# Semeval 2023 Task 3 "Detecting the Persuasion Techniques in Online News"
We only consider Subtask 3 - Persuasion Techniques
We only consider German and English subsets

Papers: https://aclanthology.org/2023.semeval-1.317/, https://aclanthology.org/D19-1565/
SemEval: https://propaganda.math.unipd.it/semeval2023task3
Annotation Guidelines: https://knowledge4policy.ec.europa.eu/sites/default/files/JRC132862_technical_report_annotation_guidelines_final_with_affiliations_1.pdf

## Setup
Run semeval23persuasion.iypnb to preprocess the dataset.

## Dataset Details
```
Articles in six languages (English, French, German, Italian, Polish, and Russian) are collected from 2020 to mid 2022, they revolve around a fixed range of widely discussed topics such as COVID-19, climate change, abortion, migration, the build-up leading to the Russo-Ukrainian war, and events related and triggered by the aforementioned war, and some country-specific local events such as elections, etc.
Our media selection covers both mainstream media and alternative news and web portals, large fraction of which were identified by fact-checkers and media credibility experts as potentially spreading mis-/disinformation.
```

## Labels
Every sentence is labeled with zero to multiple persuasion technique(s).
Label Descriptions are taken from the annotation guidelines.

{'Slogans', 'Loaded_Language', 'Appeal_to_Fear-Prejudice', 'Conversation_Killer', 'Red_Herring', 'Guilt_by_Association', 'Flag_Waving', 'o', 'Appeal_to_Hypocrisy', 'Exaggeration-Minimisation', 'Appeal_to_Authority', 'Name_Calling-Labeling', 'Causal_Oversimplification', 'False_Dilemma-No_Choice', 'Appeal_to_Popularity', 'Obfuscation-Vagueness-Confusion', 'Doubt', 'Straw_Man', 'Whataboutism', 'Repetition'}

Idea: Fine-grained and coarse graiend persuasion classification?

I need your help to summarize definitions of various persuasion techniques.
This is a persuasion technique and its definition. Please summarize it for me with two sentences:

## -- 1. Attack on the Reputation --

### Name_Calling-Labeling: 
A form of argument in which loaded labels are directed at an individual or a group, typically in an insulting or demeaning way, or as either something the target audience fears, hates, or on the contrary finds desirable or loves. This technique calls for a qualitative judgement that disregards facts and focuses solely on the essence of the subject being characterized. It is in a way also a manipulative wording, as it is used at the level of the nominal group rather than being a full-fledged argument with premise and conclusion. For example, in the political discourse, typically one is using adjectives and nouns as labels that refer to political orientation, opinions, personal characteristics, and association with some organisations, as well as insults. What distinguishes it from the Loaded Language technique, is that it is only concerned with the characterization of the subject.

Examples:
- "My opponent is a flip-flop man who cannot make up his mind. He changes mind with the breeze! How could anyone follow such a weak-willed flip-flopper?"
- "'Fascist' Anti-Vax Riot Sparks COVID Outbreak in Australia."
- "Don't get a rotten APPLE. Trust what is inside ANDROID"

### Guilt_by_Association:
Attacking the opponent or an activity by associating it with another group, activity, or concept that has sharp negative connotations for the target audience. The most common example, which has given its name to that technique, is making comparisons to Hitler and the Nazi regime. However, it is important to emphasize that this technique is not restricted to comparisons to that group only. More precisely, this can be done by claiming a link or an equivalence between the target of the technique and any individual, group or event in the presence or in the past, which has or had an unquestionable negative perception (e.g., was considered a failure), or is depicted in such a way.
NOTE: This technique is related to Whataboutism, but the latter focuses on distracting from the topic, not on attacking the opponent directly. This technique can be seen as a specific type of Casting Doubt.

Examples:
• "Do you know who else was doing that ? Hitler!”
• "Only one kind of person can think in that way: a communist.”
• "He talks like an EU official!”
• "Manohar is a big supporter for equal pay for equal work. This is the same policy that all those extreme feminist groups support. Extremists like Manohar should not be taken seriously – at least politically.”
• That company is aligned with eugenics movement.

### Doubt: 
Casting doubt on the character or the personal attributes of someone or something in order to question their general credibility or quality, instead of using a proper argument related to the topic. This can be done for instance, by speaking about the target’s professional background, as a way to discredit their argument. Casting doubt can also be done by referring to some actions or events carried out or planned by some entity that are/were not successful or appear as (probably) resulting in a failure to achieve the planned goals.

Examples:
• "A candidate talks about his opponent and says: Is he ready to be the Mayor?”
• "This task is quite complex. Is his professional background, experience and the time left sufficient to accomplish the task at hand?”
• "If you have nothing to hide, you have nothing to fear

### Appeal_to_Hypocrisy: 
The target of the technique is attacked on their reputation by charging them with hypocrisy or inconsistency. This can be done explicitly by calling out hypocrisy directly, or more implicitly by underlining the contradictions between different positions that were held or actions that were done in the past. A special way of calling out hypocrisy is by stating that someone who criticizes you for something you did, also did it in the past.

Examples:
• "How can you demand that I eat less meat to reduce my carbon footprint if you yourself drive a big SUV and fly for holidays to Bali?”
• "My parents used to speed on the highway, so they don’t have any right to tell me to slow down.”

### Questioning the Reputation
This technique is used to attack the reputation of the target by making strong negative claims about it, focusing in particular on undermining its character and moral stature rather than relying on an argument about the topic. Whether the claims are true or false is irrelevant for the effective use of this technique. Smears can be used at any point in a discussion. One particular way of using this technique is to preemptively call into question the reputation/credibility of an opponent, before he had any chance to express himself, therefore biasing the audience perception - hence one of the names of this technique is Poisoning the well. The main difference between the Casting Doubt and Questioning the Reputation technique is that the former focuses on questioning the capacity, capabilities and credibility, while the latter attempts to undermine the overall reputation, moral qualities, behaviour, etc.

Examples:
• "My opponent has a record of lying and trying to cover her dishonest dealings with a pleasant smile. Don’t let her convince you to believe her words.”
• I hope I presented my argument clearly. Now, my opponent will attempt to refute my argument by his own fallacious, incoherent, illogical version of history.

## -- 2. Justification --
Justifications consist of two parts: a statement (to propose/support or not to propose/support something) and an explanation for it, which can be an appeal to values, nationalism, popularity, fear, etc. In this context, it is of paramount importance to understand that the sole occurrence of words and phrases referring to fear, values, nationalism, etc. DOES NOT per se qualify the respective text snippet to be labeled as some type of JUSTIFICATION persuasion technique. What is being justified has also to be present in the text, even if it appears in a broader context or is implicit.

### Flag_Waving: 
Justifying or promoting an idea by extolling the pride of a group or highlighting the benefits for that specific group. The stereotypical example would be national pride, hence the name of the technique; however, the target group it applies to might be any group, e.g., related to race, gender, political preference, etc. The connection to nationalism, patriotism or benefit for an idea, group or country might be fully unfounded and is usually based on the presumption that the recipients already have certain beliefs, biases, and prejudices about the given issue. It can be seen as an appeal to emotions instead of to the logic of the audience aiming to manipulate them to win an argument. As such, this technique can also appear outside the form of well-constructed argument, by simply trying to resonate with the feeling of a particular group and as such setting up a context for further arguments.

Examples:
• "Patriotism mean no questions.”
• "Entering this war will make us have a better future in our country.”
• "We should make America great again, and restrict the immigration laws.”


### Appeal_to_Authority:
This technique gives weight to a claim or thesis by simply stating that a particular entity considered as an authority, e.g., a person or an organisation, is the source of the information. The entity mentioned as an authority may, but does not need to be an actual valid authority in the domain-specific field to discuss a particular topic or to be considered and serve as an expert. What is important, and makes it different than simply sourcing an information, is that the tone of the text indicates that it capitalises on the weight of an alleged authority in order to justify some information, claim, or conclusion. Reference to a valid authority is not a logical fallacy, a reference to an invalid authority is, and both are captured within this label. In particular, a self-reference as an authority falls under Appeal to Authority too.

Examples:
• "Richard Dawkins, an evolutionary biologist and perhaps the foremost expert in the field, says that evolution is true. Therefore, it’s true.”
• "If Napoleon said so it must be true then.” 
• "According to Serena Williams, our foreign policy is the best on Earth. So we are in the right direction.”
• Since the Pope said that this aspect of the doctrine is true we should add it to the creed.


### Appeal_to_Popularity: 
Also referred to as Bandwagon. This technique gives weight to an argument or an idea by justifying it on the basis that allegedly ‘everybody’ (or the large majority) agrees with it or ‘nobody’ disagrees with it. As such, the target audience is encouraged to adopt the same idea by considering ‘everyone else’ as an authority, and to join in and take the same course of action. Here, ‘everyone else’ might refer to the general public, key entities and actors in a certain domain, countries, etc. Analogously, an attempt to persuade the audience not to do something because ‘nobody else is taking the same action’ falls under our definition of appeal to popularity.

Examples:
• "Everyone is going to get the new smart phone when it comes out this weekend.”
• "Would you vote for Putin as president? 70% say yes.”
• "Because everyone else goes away to college, it must be the right thing to do. ”

### Appeal to Values
This technique gives weight to an idea by linking it to values seen by the target audience as positive. These values are presented as an authoritative reference in order to support or to reject an argument. Examples of such values are, for instance: tradition, religion, ethics, age, fairness, liberty, democracy, peace, transparency, etc. When such values are mentioned outside the context of a proper argument by simply using certain adjectives or nouns as a way of characterising something or someone, then, they fall under another label, namely, Loaded Language, which is a form of Manipulative Wording.

Examples:
• "We always did it according to the ten commandments.”
• "It’s standard practice to pay men more than women so we’ll continue adhering to the same standards this company has always followed.”

### Appeal_to_Fear-Prejudice: 
This technique aims to promote or reject an idea through the repulsion or fear of the audience towards this idea (e.g., by exploiting some preconceived judgements) or towards its alternative. The alternative could be the status quo, in which case the current situation is described in a scary way with Loaded Language. If the fear is linked with the consequences of a decision, it is often the case that this technique is used together with Consequential Oversimplification (see 6.2.3), and if there are only two alternatives and they are stated explicitly, then it is used together with the False Dilemma technique.

Examples:
• "We must stop those refugees as they are terrorists.”
• "If we don’t bail out the big automakers, the US economy will collapse. Therefore, we need to bail out the automakers.”
• It is a great disservice to the Church to maintain the pretense that there is nothing problematical about AL. A moral catastrophe is self-evidently underway and it is not possible honestly to deny its cause.





## -- 3. Distraction --

### Straw_Man
Also referred to as Misrepresentation of Someone's position. This technique appears to refute the opposing argument, but the real subject of the opposing argument is not addressed, but is instead replaced with a false one. Often, this technique is referred to as misrepresentation of the argument. First, a new argument is created via the covert replacement of the original argument with something that appears somewhat related, but is actually a different, distorted, exaggerated, or misrepresented version of the original proposition, which is referred to as ‘standing up a straw man’. Subsequently, the newly created ‘false’ argument (the strawman) is refuted, which is referred to as ‘knocking down a straw man’. Often, the strawman argument is created in such a way that it is easier to refute, and thus, creating an illusion of having defeated an opponent’s real proposition. Fighting a strawman is easier than fighting against a real person, which explains the origin of the name of this technique. In practice, it appears often as an abusive reformulation or explanation of what the opponent ‘actually’ means or wants.

Examples:
• "Referring to your claim that providing medicare for all citizens would be costly and a danger to the free market, I infer that you don’t care if people die from not having healthcare, so we are not going to support your endeavour."
• the corporate (i.e. private sector) players in global governance are determined to have their agenda accepted everywhere — which is none other than to grant themselves full powers over the planet.

### Red_Herring (Introducing Irrelevant Information): 
This technique consists in diverting the attention of the audience from the main topic being discussed by introducing another topic. The aim of attempting to redirect the argument to another issue is to focus on something the person doing the redirecting can better respond to or to leave the original topic unaddressed. The name of that technique comes from the idea that a fish with a strong smell (like a herring) can be used to divert dogs from the scent of a prey they are following. A strawman (defined earlier) is also a specific type of a red herring in the way that it distracts from the main issue by painting the opponent’s argument in an inaccurate light.

Examples:
• "I have worked hard to help eliminate criminal activity. What we need is economic growth that can only come from the hands of leadership."
• "Lately, there has been a lot of criticism regarding the quality of our product. We’ve decided to have a new sale in response, so you can buy more at a lower cost!"

### Whataboutism (Switching Topic)
A technique that attempts to discredit an opponent’s position by charging them with hypocrisy without directly disproving their argument. Instead of answering a critical question or argument, an attempt is made to retort with a critical counter-question which expresses a counter-accusation, e.g., mentioning double standards, etc. The intent is to distract from the content of a topic and actually switch the topic. 

Examples:
• "A nation deflects criticism of its recent human rights violations by pointing to the history of slavery in the United States."
• "Qatar spending profusely on Neymar, not fighting terrorism."



## -- 4. Simplification --

### Causal_Oversimplification: 
Assuming a single cause or reason when there are actually multiple causes for an issue. This technique has the following logical form(s): "Y occurred after X; therefore, X was the only cause of Y", "X caused Y; therefore, X was the only cause of Y" (although A,B,C...etc. also contributed to Y.)

Examples:
• "President Trump has been in office for a month and gas prices have been skyrocketing. The rise in gas prices is because of President Trump."
• "School violence has gone up and academic performance has gone down since video games featuring violence were introduced. Therefore, video games with violence should be banned, resulting in school improvement."


### False_Dilemma-No_Choice: 
Sometimes called the “either-or” fallacy, a false dilemma is a logical fallacy that presents only two options or sides when there are many options or sides. In extreme cases, the authors tells the audience exactly what actions to take, eliminating any other possible choices (hence the label Dictatorship). This technique has the following logical form: (a) Black & White Fallacy: There are only two alternatives A and B to a given problem/task. It cannot be A. Therefore, the only solution is B (since A is not an option). (b) Dictatorship The only solution to a given problem/task is A.

Examples:
• "Either we raise new taxes, or the roads will become unusable.”
• "There is no alternative to Pfizer Covid-19 vaccine."

### Consequential Oversimplification (Slippery Slope)
An argument/idea is rejected and instead of discussing whether it makes sense and/or is valid, the argument affirms, without proof, that accepting the proposition would imply accepting other propositions that are considered negative. This technique has the following logical form: if A will happen then B, C, D, ... will happen In the above definition: • A is something one is trying to REJECT • B, C, D are perceived as some potential negative consequences happening if A happens. The core essence behind the ‘Slippery Slope’ (Consequential Oversimplification) is an assertion one is making of some ‘first’ event/action leading to a domino-like chain of events that have some significant negative effects and consequences that appear to be ludicrous. The slippery slope is characterized by ignoring and/or understating the likelihood of the sequence of events from the first event leading to the end point (last event) of the slope.

Examples (Rejection):
• "If we allow same-sex marriage, we will soon see marriage between siblings or even marriages between humans and animals!”
• "If we let our government ban certain guns, they will eventually ban all guns.”
• If we legalize pot, then that will lead to every drug in the world becoming legal.
• Today, women want the vote. Tomorrow, they’ll want to be doctors and lawyers, and then combat soldiers

Examples (Support):
• "If we stop buying gas from Russia, Russia will go bankrupt, and this will lead to the end of the war in Ukraine, which will start the process of Ukraine joining NATO ..."
• "If only the European leaders could take the fate of the European countries into their own hands, instead of letting them drift into the Atlantist swamps, the conflict would end immediately"

## -- 5. CALL -- 

### Slogans:
A brief and striking phrase that may include labeling and stereotyping. Slogans tend to act as emotional appeals.

Examples:
• "Our "unity in diversity" contrasts with the divisions everywhere else.”
• "Make America great again!"
• "Immigrants welcome, racist not!", "No border. No control!"

### Conversation_Killer: 
Also referred to as Thought-terminating Cliché. Words or phrases that discourage critical thought and meaningful discussion about a given topic are considered as Conversetional killers. They are a form of loaded language, often passing as folk wisdom, intended to end an argument and quell cognitive dissonance.

Examples:
• "Just Say No.”
• "That’s just your opinion."
• "You can’t change human nature."

### Appeal to Time (Kairos)
The argument is centred around the idea that time has come for a particular action. The very timeliness of the idea is part of the argument. The call to “Act Now!” is an example of Appeal to Time.

Examples:
• "If majority of the population does not get vaccinated within a month the pandemic will kill us! So, we need to start vaccinations right now.”
• "This is no time to engage in the luxury of cooling off or to take the tranquilizing drug of gradualism. Now is the time to make real the promises of democracy. Now is the time to rise from the dark and desolate valley of segregation to the sunlit path of racial justice." [Martin Luther King, 1963]
• Should we vaccinate the entire population in the context of the current pandemic? This is the right time to do it, and this is the right thing.

## -- 6. Manipulative Wording --

### Loaded_Language: 
This fallacy uses specific words and phrases with strong emotional implications (either positive or negative) to influence and to convince the audience that an argument is valid/true. It is also known as appeal to/argument from emotive language.

Examples:
• "How stupid and petty things have become in Washington”
• "They keep feeding these people with trash. They should stop.”


### Obfuscation-Vagueness-Confusion: 
This fallacy uses words that are deliberately not clear so that the audience may have its own interpretations. For example, an unclear phrase with multiple or unclear definitions is used within the argument and, therefore, does not support the conclusion. Statements that are imprecise and intentionally do not fully or vaguely answer the question posed fall under this category too.

Examples:
• "It is a good idea to listen to victims of theft. Therefore if the victims say to have the thief shot, then you should do that.” ["listen to" is equivocated here]
• "We will hex-develop the blockchain with AI-based interconnectors to maximize ROI.” [use of nonsense words]
• “Feathers can not be dark, because all feathers are light!”
• The significance of the passage of time, right? The significance of the passage of time. So when you think about it, there is great significance to the passage of time.


### Exaggeration-Minimisation: 
This fallacy consists of either representing something in an excessive manner – making things larger, better, worse (e.g., ‘the best of the best’, ‘quality guaranteed’) – or making something seem less important or smaller than it really is (e.g., saying that an insult was just a joke), downplaying statements and ignoring arguments and accusations made by an opponent.

Examples:
• "Democrats bolted as soon as Trump’s speech ended in an apparent effort to signal they can’t even stomach being in the same room as the president.”
• "Why did you fight her? I was not fighting with her; we were just playing.”
• From the seminaries, to the clergy, to the bishops, to the cardinals, homosexuals are present at all levels, by the thousand.


### Repetition
The speaker uses the same word, phrase, story, or imagery repeatedly with the hopes that the repetition will lead to persuading the audience.

Examples:
• "Hurtlocker deserves an Oscar. Other films have potential, but they do not deserve an Oscar like Hurtlocker does. The other movies may deserve an honorable mention but Hurtlocker deserves the Oscar.”
• "Stupid people are so annoying. They prove their stupidity by saying stupid things.”


## -- 6. o -- 
Nothing 
