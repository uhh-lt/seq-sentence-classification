import random
from typing import List, Dict, Optional


def get_class_name(rawtag):
    # get (finegrained) class name
    if rawtag.startswith("B-") or rawtag.startswith("I-"):
        return rawtag[2:]
    else:
        return rawtag


class FewshotSampleBase:
    """
    Abstract Class
    DO NOT USE
    Build your own Sample class and inherit from this class
    """

    def __init__(self):
        self.class_count = {}

    def get_class_count(self) -> Dict[str, int]:
        """
        return a dictionary of {class_name:count} in format {str : int}
        """
        return self.class_count


class Sample(FewshotSampleBase):
    def __init__(
        self,
        sentences: List[str],
        tags: List[str],
        other_tags: List[str] = ["o"],
        is_span_annotation: bool = True,
    ):
        self.is_span_annotation = is_span_annotation
        self.tags = [tag.lower() for tag in tags]
        self.sentences = [sentence.lower() for sentence in sentences]
        # strip B-, I-
        self.normalized_tags = list(map(get_class_name, self.tags))
        self.class_count = {}
        self.other_tags = other_tags

    def __count_entities__(self):
        if self.is_span_annotation:
            current_tag = self.normalized_tags[0]
            for tag in self.normalized_tags[1:]:
                if tag == current_tag:
                    continue
                else:
                    if current_tag not in self.other_tags:
                        if current_tag in self.class_count:
                            self.class_count[current_tag] += 1
                        else:
                            self.class_count[current_tag] = 1
                    current_tag = tag
            if current_tag not in self.other_tags:
                if current_tag in self.class_count:
                    self.class_count[current_tag] += 1
                else:
                    self.class_count[current_tag] = 1
        else:
            # just count every occurrence of tag in normalized_tags
            self.class_count = {
                tag: self.normalized_tags.count(tag)
                for tag in set(self.normalized_tags)
                if tag not in self.other_tags
            }

    def get_class_count(self):
        if self.class_count:
            return self.class_count
        else:
            self.__count_entities__()
            return self.class_count

    def get_tag_class(self):
        # strip 'B' 'I'
        tag_class = list(set(self.normalized_tags))
        for other_tag in self.other_tags:
            if other_tag in tag_class:
                tag_class.remove(other_tag)
        return tag_class

    def valid(self, target_classes):
        return (
            set(self.get_class_count().keys()).intersection(set(target_classes))
        ) and not (set(self.get_class_count().keys()).difference(set(target_classes)))

    def __str__(self):
        newlines = zip(self.sentences, self.tags)
        return "\n".join(["\t".join(line) for line in newlines])


class FewshotSampler:
    """
    sample one support set and one query set
    """

    def __init__(
        self,
        N: int,
        K: int,
        samples: List[Sample],
        classes: Optional[List[str]] = None,
        threshold: int = 2,
        random_state: int = 0,
        must_include: Optional[int] = None,
    ):
        """
        N: int, how many types in each set
        K: int, how many instances for each type in support set
        samples: List[Sample], Sample class must have `get_class_count` attribute
        classes[Optional]: List[any], all unique classes in samples. If not given, the classes will be got from samples.get_class_count()
        random_state[Optional]: int, the random seed
        """
        self.must_include = must_include
        self.threshold = threshold
        self.K = K
        self.N = N
        self.samples = samples
        self.__check__()  # check if samples have correct types
        if classes:
            self.classes = classes
        else:
            self.classes = self.__get_all_classes__()
        random.seed(random_state)

    def __get_all_classes__(self) -> List[str]:
        classes = []
        for sample in self.samples:
            classes += list(sample.get_class_count().keys())
        return list(set(classes))

    def __check__(self) -> None:
        for idx, sample in enumerate(self.samples):
            if not hasattr(sample, "get_class_count"):
                print(
                    f"[ERROR] samples in self.samples expected to have `get_class_count` attribute, but self.samples[{idx}] does not"
                )
                raise ValueError

    def __additem__(self, index: int, set_class: Dict[str, int]) -> None:
        class_count = self.samples[index].get_class_count()
        for class_name in class_count:
            if class_name in set_class:
                set_class[class_name] += class_count[class_name]
            else:
                set_class[class_name] = class_count[class_name]

    def __valid_sample__(
        self,
        sample: FewshotSampleBase,
        set_class: Dict[str, int],
        target_classes: List[str],
    ) -> bool:
        threshold = self.threshold * set_class["k"]
        class_count = sample.get_class_count()
        if not class_count:
            return False
        isvalid = False
        for class_name in class_count:
            if class_name not in target_classes:
                return False
            if class_count[class_name] + set_class.get(class_name, 0) > threshold:
                return False
            if set_class.get(class_name, 0) < set_class["k"]:
                isvalid = True
        return isvalid

    def __finish__(self, set_class: Dict[str, int]) -> bool:
        if len(set_class) < self.N + 1:
            return False
        for k in set_class:
            if set_class[k] < set_class["k"]:
                return False
        return True

    def __get_candidates__(self, target_classes: List[str]) -> List[int]:
        return [
            idx
            for idx, sample in enumerate(self.samples)
            if sample.valid(target_classes)
        ]

    def __next__(self):
        """
        randomly sample one support set and one query set
        return:
        target_classes: List[any]
        support_idx: List[int], sample index in support set in samples list
        """
        support_class = {"k": self.K}
        support_idx = []
        target_classes = random.sample(self.classes, self.N)
        candidates = self.__get_candidates__(target_classes)
        while not candidates:
            target_classes = random.sample(self.classes, self.N)
            candidates = self.__get_candidates__(target_classes)

        if self.must_include:
            self.__additem__(self.must_include, support_class)
            support_idx.append(self.must_include)

        # greedy search for support set
        while not self.__finish__(support_class):
            index = random.choice(candidates)
            if index not in support_idx:
                if self.__valid_sample__(
                    self.samples[index], support_class, target_classes
                ):
                    self.__additem__(index, support_class)
                    support_idx.append(index)

        return target_classes, support_idx, support_class

    def __iter__(self):
        return self
