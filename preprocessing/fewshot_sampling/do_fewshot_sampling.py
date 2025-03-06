import pandas as pd
from typing import List, Optional, Union
from fewshotsampler import Sample, FewshotSampler
from pathlib import Path
import concurrent.futures
from typing import TypedDict


class Parameters(TypedDict):
    N: int
    K: int
    threshold: int
    random_state: int


train_paths = [
    Path("../../datasets/csabstruct/train.parquet"),
    # Path("../../datasets/pubmed200k/train.parquet"),
    # Path("../../datasets/coarsediscourse/coursediscourse_train.parquet"),
    # Path("../../datasets/daily_dialog/dailydialog_train.parquet"),
    # Path("../../datasets/emotion_lines/friends_train.parquet"),
    # Path("../../datasets/wikisection/en/city/wikisection_en_city_train.parquet"),
    # Path("../../datasets/wikisection/de/city/wikisection_de_city_train.parquet"),
]

for train_path in train_paths:
    df = pd.read_parquet(train_path)
    print(f"{train_path}: Number of samples: {len(df)}")

other_tags_list = [
    ["o"],  
    # ["o"],
    # ["other"],
    # ["neutral"],
    # ["neutral", "non-neutral"],
    # ["o"],
    # ["o"],
]

is_span_annotation_list = [
    True,
    # True,
    # False,
    # False,
    # False,
    # ["o"],
    # ["o"],
]

the_must_include_ids: List[Union[None, int]] = [None]
# [218, 438]


def find_fewshot_samples(
    samples: List[Sample], params: Parameters, must_include_id: Optional[int]
):
    # this could take forever, depending on the params
    sampler = FewshotSampler(
        N=params["N"],
        K=params["K"],
        samples=samples,
        threshold=params["threshold"],
        random_state=params["random_state"],
        must_include=must_include_id,
    )
    return sampler.__next__()


def try_seeds_and_thresholds(
    samples: List[Sample],
    seeds: List[int],
    thresholds: List[int],
    N: int,
    K: int,
    timeout=2,
    must_include_ids: List[Union[None, int]] = [None],
):
    results = []
    result_parameters = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for threshold in thresholds:
            print("Trying threshold", threshold)

            for seed in seeds:
                print("Trying seed", seed)  #

                for must_include_id in must_include_ids:
                    try:
                        params: Parameters = {
                            "N": N,
                            "K": K,
                            "threshold": threshold,
                            "random_state": seed,
                        }
                        future = executor.submit(
                            find_fewshot_samples, samples, params, must_include_id
                        )
                        result = future.result(timeout=timeout)
                        # print(f"Success: with seed {seed}, threshold {threshold}")
                        results.append(result)
                        result_parameters.append(params)
                    except concurrent.futures.TimeoutError:
                        # print(f"Timeout with seed {seed}, threshold {threshold}")
                        future.cancel()
                    except Exception as e:
                        print(
                            f"Exception with seed {seed}, threshold: {threshold}: {e}"
                        )

            if len(results) > 0:
                break

        executor.shutdown(wait=False, cancel_futures=True)

    print("Found", len(results), "results")
    return results, result_parameters


def find_best_result(results: List, result_parameters: List[Parameters]):
    if len(results) == 0:
        return None, None, None

    # the best results are those that have the smalles threshold
    # first step is to find the smalles threshold
    thresholds = [params["threshold"] for params in result_parameters]
    min_threshold = min(thresholds)

    # now we need to find the results that have the smallest threshold
    results_with_min_threshold = [
        result
        for result, params in zip(results, result_parameters)
        if params["threshold"] == min_threshold
    ]

    # now we need to find the result with the smallest number of samples
    min_samples = float("inf")
    best_sample_indices = None
    best_class_counts = None
    best_parameters = None

    for i, result in enumerate(results_with_min_threshold):
        _, sample_indices, sample_class_counts = result
        num_samples = len(sample_indices)
        if num_samples < min_samples:
            min_samples = num_samples
            best_sample_indices = sample_indices
            best_class_counts = sample_class_counts
            best_parameters = result_parameters[i]

    return best_sample_indices, best_class_counts, best_parameters


def store_best_result(
    samples: List[Sample],
    best_sample_indices,
    best_sample_classes,
    best_params: Parameters,
    output_path: Path,
):
    # Select the samples with the indices in min_idx
    few_shot_samples = [samples[idx] for idx in best_sample_indices]

    # Convert the selected samples to a DataFrame
    few_shot_df = pd.DataFrame(
        [
            {"sentences": sample.sentences, "labels": sample.tags}
            for sample in few_shot_samples
        ]
    )

    # Save the DataFrame to a parquet file
    few_shot_df.to_parquet(
        output_path.with_name(f'few_shot_nAll_k{best_params["K"]}.parquet')
    )

    # Save some statistics to a text file
    with open(
        output_path.with_name(f'few_shot_nAll_k{best_params["K"]}.txt'), "w"
    ) as f:
        f.write(f"Number of selected samples: {len(few_shot_samples)}\n")
        f.write(f"Class counts: {best_sample_classes}\n")
        f.write(f"Support samples: {best_sample_indices}\n")
        f.write(f'N: {best_params["N"]}\n')
        f.write(f'K: {best_params["K"]}\n')
        f.write(f'Threshold: {best_params["threshold"]}\n')


K = 16
seeds = [
    42,
    420,
    1337,
    69,
    96,
    555,
    666,
    777,
    88,
    7,
    1,
    3,
    1995,
    1996,
    101,
    404,
    2025,
    52,
    66,
    9000,
    32,
    64,
    86400,
    21,
    911,
    50,
    100,
    720,
    1234,
    4321,
]
thresholds = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

for train_path, other_tags, is_span_annotation in zip(
    train_paths, other_tags_list, is_span_annotation_list
):
    print(f"----- Processing {train_path} ----- ")

    # load the samples
    df = pd.read_parquet(train_path)
    samples = [
        Sample(
            sentences,
            tags,
            is_span_annotation=is_span_annotation,
            other_tags=other_tags,
        )
        for sentences, tags in zip(df["sentences"], df["labels"])
    ]

    # count the number of unique lables in this dataset
    unique_labels = set()
    for sample in samples:
        unique_labels.update(sample.tags)

    # if other_tag is in the unique labels, remove it
    for other_tag in other_tags:
        if other_tag in unique_labels:
            unique_labels.remove(other_tag)

    num_classes = len(unique_labels)
    print(f"Number of samples: {len(samples)}")
    print("Unique labels:", unique_labels)
    print(f"Number of unique labels: {num_classes}")

    print(f"----- START {train_path} ----- ")

    # find fewshot samples
    results, result_parameters = try_seeds_and_thresholds(
        samples,
        seeds,
        thresholds,
        N=num_classes,
        K=K,
        must_include_ids=the_must_include_ids,
    )
    best_sample_indices, best_sample_classes, best_params = find_best_result(
        results, result_parameters
    )

    if (
        best_sample_indices is None
        or best_sample_classes is None
        or best_params is None
    ):
        print(f"Could not find fewshot samples for {train_path}")
    else:
        store_best_result(
            samples, best_sample_indices, best_sample_classes, best_params, train_path
        )
        print(
            f"Found&Saved fewshot samples for {train_path}, threshold: {best_params['threshold']}, seed: {best_params['random_state']}"
        )

    print(f"----- END   {train_path} ----- ")
    print()
