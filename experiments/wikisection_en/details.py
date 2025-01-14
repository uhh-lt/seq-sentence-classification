project_name = "Topic Analysis of Wikipedia Articles about Cities"

project_details = "assigning each sentence in the document a corresponding section class in which the sentence appears."

label_dict = {
    "transport": "Transport refers to the systems and infrastructure that enable the movement of people and goods from one location to another. This encompasses various modes of transportation, including roadways, railways, airways, waterways and other.",
    "people": "The category people refers to individuals who have a significant connection to a particular place, often due to their birth, residence, or notable contributions. These individuals can encompass a wide range of professions, backgrounds and accomplishments.",
    "architecture": "The category architecture encompasses the design and construction of buildings and other physical structures. It includes a wide range of styles, periods, and purposes, reflecting both aesthetic and functional considerations.",
    "health": "Health refers to the description of the healthcare system and resources available to residents of a particular area. This includes hospitals, clinics, and other healthcare facilities, as well as public health programs and initiatives.",
    "culture": "Culture refers to the shared practices, expressions, values, and beliefs that define a community or society. It encompasses the ways in which people live, interact, and make sense of the world around them.",
    "economics": "Economics refers to the description of the production, distribution, and consumption of goods and services within a specific geographic area. It encompasses the various economic activities, resources, and conditions that shape the financial well-being of a region.",
    "etymology": "Etymology refers to the investigation of the origin and historical development of words, especially place names. It involves tracing the linguistic roots and evolution of words to understand their meaning and how they have changed over time. ",
    "tourism": "Tourism refers to the activities of people traveling to and staying in places outside their usual environment for leisure, business, or other purposes. It encompasses the attractions, activities, and infrastructure that cater to the needs and interests of visitors.",
    "facility": "Facility refers to a place, structure, or system that is designed to serve a specific purpose or provide a particular service to a community. This includes infrastructure for essential services, public buildings, commercial establishments, and other places that support the needs of the population.",
    "education": "Education refers to the institutions, programs, and activities involved in the process of imparting knowledge, skills, and values to individuals. This includes formal schooling at all levels, as well as informal learning opportunities and educational resources available in a community.",
    "science": "Science refers to the systematic study of the natural world through observation, experimentation, and analysis. In the context of a location, it encompasses the scientific institutions, research activities, and natural phenomena that contribute to our understanding of the world.",
    "history": "History refers to the record of past events and developments that have shaped a particular location or region. It encompasses the chronological narrative of the place, including its origins, major events, cultural shifts, and other noteworthy occurrences that have contributed to its present-day character.",
    "law": "Law refers to the system of rules and regulations that govern conduct and maintain order within a society. In the context of a specific location, it encompasses the law enforcement agencies, legal institutions, and community safety initiatives that work together to uphold the law and ensure public safety.",
    "faith": "Faith refers to the belief in and worship of a supernatural power or deity, or an overarching system of beliefs, values, and practices to which people adhere. In the context of a location, it encompasses the various religions, denominations, and spiritual traditions that are practiced by the people, as well as the institutions and leaders that support their faith.",
    "demography": "Demography refers to the statistical study of human populations, especially with reference to size, density, distribution, and vital statistics (births, 1  deaths, marriages, etc.). It provides a quantitative understanding of the characteristics and dynamics of populations.",
    "other": "Other is a miscellaneous category that encompasses information relevant to the place but not fitting into any of the other defined categories. It serves as a repository for interesting facts, trivia, and unique details that enrich the overall understanding of the location.",
    "media": "Media refers to the various channels of communication that are used to disseminate news, information, entertainment, and other content to a wide audience. This includes print media, broadcast media, and online media, as well as the organizations and individuals involved in producing and distributing this content.",
    "geography": "Geography refers to the study of the Earth's physical features, climate, and human activity as it affects and is affected by these, including the distribution of populations and resources, land use, and industries. In the context of a specific location, it describes the place's position on Earth, its natural environment, and its relationship to surrounding areas.",
    "sights": "Sights refers to places, landmarks, or objects that are of visual, historical, or cultural interest. They are noteworthy features of a location that attract attention, provide aesthetic pleasure, or offer insights into the area's past, present, or cultural identity.",
    "international_affairs": "International_affairs refers to the interactions and relationships between a location (such as a city, region, or country) and other countries or international entities. It encompasses a wide range of activities, including diplomacy, trade, cultural exchange, and cooperation on global issues.",
    "overview": "Overview refers to a concise and general description of a location, highlighting its key features, history, culture, and other noteworthy aspects. It provides a broad understanding of the place and its unique characteristics.",
    "infrastructure": "Infrastructure refers to the underlying foundation or basic framework of a system or organization. In the context of a city or region, it encompasses the essential physical and organizational structures and facilities that enable it to function effectively, including transportation networks, utilities, communication systems, and public services.",
    "district": "District refers to a defined area or division within a larger geographic entity, often with a specific function, character, or purpose. Districts can be administrative units, historical neighborhoods, cultural zones, or areas with distinct economic activities.",
    "environment": "Environment refers to the natural world and its surroundings, including the living and non-living components that interact to create the conditions for life. In the context of a specific location, it encompasses the climate, natural resources, environmental issues, and human activities that impact the natural world.",
    "society": "Society refers to the aggregate of people living together in a more or less ordered community. In the context of a specific location, it encompasses the social structures, institutions, and shared values that shape the lives and interactions of its inhabitants.",
    "climate": "Climate refers to the long-term average weather patterns in a particular region or area. It includes factors such as temperature, precipitation, humidity, wind, and sunshine, as well as their seasonal variations and any characteristic weather events.",
    "politics": "Politics refers to the activities associated with the governance of a country or area, especially the debate or conflict among individuals or parties having or hoping to achieve 1  power. In the context of a specific location, it encompasses the political systems, processes, and institutions that shape the exercise of power and decision-making",
    "recreation": "Recreation refers to activities done for enjoyment when one is not working. In the context of a specific location, it encompasses the places, facilities, and opportunities that allow people to engage in leisure activities, relax, and improve their physical and mental well-being.",
    "sport": "Sport refers to an activity involving physical exertion and skill in which an individual or team competes against another or others for entertainment. 1  In the context of a specific location, it encompasses the various sports, teams, facilities, and events that are part of the community's recreational and competitive landscape.",
    "crime": "Crime refers to an action or omission that constitutes an offense that may be prosecuted by the state and is punishable by law. 1  In the context of a specific location, it encompasses the occurrence of criminal activities, law enforcement efforts to address crime, and the overall level of public safety and security."
}

system_prompt_template = """
You are a professional annotator specialized in annotating sentences with the help of annotation guidelines.
You strictly adhere to the guidelines and follow the desired output format.
You are a member of the project <project_name> which is about <project_details>.

Annotation Guidelines:
<annotation_guidelines>

Output Format:
You MUST answer in this JSON format, but the reason is optional:
[
    {
        "text_id": 1,
        "reason": "This sentence is typical for the transport section.",
        "category": "transport"
    },
    {
        "text_id": 2,
        "category": "health"
    },
    {
        "text_id": 3,
        "reason": "This sentence describes economic activities.",
        "category": "economics"
    },
    ...
]
"""

user_prompt_template = """
Please annotate each sentence of the following document.

Document:
{document}

Remember to annotate every provided sentence. You MUST use the categories provided in the Annotation Guidelines!
"""