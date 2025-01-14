project_name = "Themenanalyse von Wikipedia-Artikeln über Städte"

project_details = "jedem Satz im Dokument eine entsprechende Abschnittsklasse zuzuweisen, in der der Satz erscheint."

label_dict = {
    "architektur": "Architektur umfasst Beschreibungen von Bauwerken und deren Umgebung. Dies beinhaltet sowohl die ästhetischen und stilistischen Merkmale der Bauwerke selbst als auch ihre historische Bedeutung, Funktionalität und räumliche Einordnung.",
    "bildung": "Bildung bezieht sich auf die Beschreibung von Institutionen und Einrichtungen, die der Vermittlung von Wissen und Fähigkeiten dienen. Dies umfasst verschiedene Bildungsniveaus, von der Grundschule bis zur Universität, sowie die unterschiedlichen Fachbereiche und Spezialisierungen.",
    "demografie": "Demografie bezieht sich auf die statistische Beschreibung einer Bevölkerung. Dies umfasst die Erfassung und Analyse von Daten wie Bevölkerungszahl, -dichte, -zusammensetzung und -entwicklung. Demografische Daten liefern Informationen über die Struktur und Dynamik einer Bevölkerung und sind relevant für verschiedene Bereiche wie Politik, Wirtschaft und Sozialwissenschaften.",
    "erholung": "Erholung bezieht sich auf die Beschreibung von Möglichkeiten, die Menschen nutzen können, um sich von der Arbeit und dem Alltag zu erholen und ihre Freizeit zu genießen. Dazu gehören sowohl natürliche Umgebungen als auch von Menschen geschaffene Einrichtungen und Aktivitäten.",
    "etymologie": "Etymologie bezieht sich auf die Erklärung der Herkunft und Entwicklung von Namen. Sie befasst sich mit der sprachlichen Analyse und historischen Rekonstruktion von Wörtern, um ihre ursprüngliche Bedeutung und Veränderungen im Laufe der Zeit zu verfolgen. Im Kontext von Ortsnamen liefert die Etymologie Einblicke in die Geschichte, Kultur und sprachliche Vielfalt einer Region.",
    "gemeinde": "Gemeinde bezieht sich auf die Beschreibung der kleinsten selbstverwalteten Einheit in einem staatlichen Gebiet. Sie umfasst die Zusammensetzung aus einzelnen Ortsteilen, die administrative Zugehörigkeit und die historische Entwicklung der Gemeinde sowie Wappen und Symbole, die sie repräsentieren.",
    "gemeindepartnerschaft": "Gemeindepartnerschaft bezieht sich auf die formelle Beziehung zwischen zwei Gemeinden in verschiedenen Ländern oder Regionen, die auf einem Partnerschaftsvertrag basiert. Ziel ist der kulturelle, soziale, wirtschaftliche und oft auch politische Austausch zwischen den Bürgern und Institutionen der Partnergemeinden.",
    "geographie": "Geographie bezieht sich auf die Beschreibung der erdkundlichen Merkmale eines Ortes. Dies umfasst die Lage, die physische Beschaffenheit (Landschaft, Gewässer), die klimatischen Bedingungen und die Beziehung zu anderen Orten.",
    "geschichte": "Geschichte bezieht sich auf die Beschreibung der Vergangenheit eines Ortes oder einer Region. Sie umfasst die chronologische Darstellung von Ereignissen, Entwicklungen und Veränderungen, die den Ort zu dem gemacht haben, was er heute ist.",
    "infrastruktur": "Infrastruktur bezieht sich auf die Beschreibung der materiellen und immateriellen Einrichtungen, die die Grundlage für das Funktionieren einer Gesellschaft bilden. Dazu gehören Anlagen und Systeme der Versorgung, des Verkehrs, der Kommunikation und der öffentlichen Dienste.",
    "kirche": "Kirche bezieht sich auf die Beschreibung von Gebäuden, die für religiöse Zwecke genutzt werden, sowie auf die damit verbundenen Institutionen und Gemeinschaften. Dies umfasst die architektonischen, historischen und religiösen Aspekte der Kirche sowie ihre Bedeutung für die lokale Gemeinschaft.",
    "klima": "Klima bezieht sich auf die Beschreibung des durchschnittlichen Wettergeschehens an einem bestimmten Ort über einen längeren Zeitraum (mindestens 30 Jahre). Es umfasst die typischen Werte und Schwankungen von Temperatur, Niederschlag, Sonnenscheindauer, Luftfeuchtigkeit und anderen Wetterelementen sowie die Ausprägung der Jahreszeiten.",
    "kriminalität": "Kriminalität bezieht sich auf die Beschreibung des Auftretens von Straftaten in einem bestimmten Gebiet. Dies umfasst die Erfassung und Analyse von Daten über die Häufigkeit und Art verschiedener Straftaten sowie die Beschreibung von besonderen Ereignissen mit kriminellem Hintergrund.",
    "kultur": "Kultur bezieht sich auf die Beschreibung der kulturellen Ausdrucksformen, Einrichtungen und Aktivitäten, die das Leben der Menschen in einem bestimmten Gebiet prägen. Dazu gehören Sprache, Bildung, Kunst, Traditionen, historische und kulturelle Stätten sowie kulturelle Einrichtungen.",
    "menschen": "Menschen bezieht sich auf die Beschreibung von Personen, die eine Verbindung zu einem bestimmten Ort haben. Dies umfasst ihre Lebensdaten, ihren Beruf oder ihre Tätigkeit, ihren Bezug zum Ort und ihre Bedeutung im lokalen, regionalen oder internationalen Kontext.",
    "politik": "Politik bezieht sich auf die Beschreibung der politischen Organisation und der politischen Prozesse in einem bestimmten Gebiet. Dies umfasst die politischen Strukturen, die politischen Akteure, die Wahlen, die Regierung und Verwaltung sowie die politischen Themen und Herausforderungen.",
    "presse": "Die Kategorie Presse umfasst Beschreibungen von Medien, die Informationen und Nachrichten verbreiten, sowie von Unternehmen, die in der Medienbranche tätig sind.",
    "regierung": "Regierung bezieht sich auf die Beschreibung der Institutionen und Personen, die die politische Macht in einem bestimmten Gebiet ausüben. Dies umfasst die Wahlen, die politischen Ämter, die Verwaltungsgliederung, die politischen Prozesse sowie die Aufgaben und Verantwortlichkeiten der Regierung.",
    "religion": "Die Kategorie Religion umfasst Beschreibungen der religiösen Gemeinschaften, Institutionen und Praktiken in einem bestimmten Gebiet.",
    "sport": "Die Kategorie Sport umfasst Beschreibungen von Sportarten, Sportstätten, Sportvereinen und Sportereignissen in einem bestimmten Gebiet.",
    "stadtlandschaft": "Stadtlandschaft bezieht sich auf die Beschreibung des visuellen Erscheinungsbildes und der räumlichen Struktur einer Stadt. Sie umfasst die Gebäude, Straßen, Plätze, die Infrastruktur, die Vegetation und andere Elemente, die das Bild der Stadt prägen.",
    "stadtviertel": "Stadtviertel bezieht sich auf die Beschreibung eines abgegrenzten Teils einer Stadt, der sich durch bestimmte gemeinsame Merkmale von anderen Stadtteilen unterscheidet. Diese Merkmale können historischer, funktionaler, architektonischer oder sozialer Natur sein.",
    "tourismus": "Tourismus bezieht sich auf die Beschreibung von Attraktionen, Einrichtungen und Aktivitäten, die Besucher in eine bestimmte Region locken. Dies umfasst Sehenswürdigkeiten, die touristische Infrastruktur und die Möglichkeiten zur Freizeitgestaltung.",
    "überblick": "Überblick bezieht sich auf die Darstellung von kurzen, interessanten Informationen über einen Ort, die nicht in die anderen Kategorien passen. Diese Informationen können aus verschiedenen Bereichen stammen und dienen dazu, das Bild des Ortes zu ergänzen und zu vervollständigen.",
    "verkehr": "Verkehr bezieht sich auf die Beschreibung der Möglichkeiten, sich in und um einen bestimmten Ort fortzubewegen. Dies umfasst die Verkehrswege, die Verkehrsmittel, die Verkehrseinrichtungen und die Verkehrsanbindung an andere Orte.",
    "wirtschaft": "Wirtschaft bezieht sich auf die Beschreibung der wirtschaftlichen Aktivitäten und Strukturen in einem bestimmten Gebiet. Dies umfasst die vorherrschenden Wirtschaftszweige, die wichtigsten Unternehmen, die verfügbaren Ressourcen und die Situation auf dem Arbeitsmarkt.",
    "sonstiges": "Sonstiges ist eine Kategorie für Informationen, die für den Ort relevant sind, aber nicht eindeutig einer der anderen Kategorien zugeordnet werden können. Sie dient als Sammelbecken für verschiedene Arten von Informationen, die das Bild des Ortes ergänzen.",
}

system_prompt_template = """
Du bist ein professioneller Annotator, der auf das Annotieren von Sätzen mit Hilfe von Annotierungsrichtlinien spezialisiert ist.
Du hältst dich strikt an die Richtlinien und befolgst das gewünschte Ausgabeformat.
Du bist Mitglied des Projekts <project_name>, mit der Aufgabe <project_details>

Annotationsrichtlinien:
<annotation_guidelines>

Ausgabeformat:
Du MUSST in diesem JSON Format antworten, aber reason ist optional:
[
    {
        "text_id": 1,
        "reason": "Dieser Satz beschreibt ist typisch für eine Sektion über Infrastruktur.",
        "category": "infrastruktur"
    },
    {
        "text_id": 2,
        "category": "infrastruktur"
    },
    {
        "text_id": 3,
        "reason": "Dieser Satz beschreibt wirtschaftliche Aktivitäten.",
        "category": "wirtschaft"
    },
    ...
]
"""

user_prompt_template = """
Bitte annotiere jeden Satz des folgenden Dokuments.

Dokument:
{document}

Denke daran, jeden Satz zu annotieren. Du MUSST die in den Annotierungsrichtlinien angegebenen Kategorien verwenden!
"""