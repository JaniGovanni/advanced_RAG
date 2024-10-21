import re
from typing import List, Optional
from unstructured.documents.elements import Element

unwanted_categories_default = ['Header', 'Footer', 'Image', 'FigureCaption', 'Formula', 'Table']


unwanted_titles_list_default_regex = [
    r"^(inhaltsverzeichnis|table\sof\scontents|toc|verzeichnis\sder\sinhalte|contents)$",
    r"^(literaturverzeichnis|quellenangaben|bibliography|references)$",
    r"^(abbildungsverzeichnis|tabellenverzeichnis)$",
    r"^(list\sof\sfigures|list\sof\stables)$"]

unwanted_titles_list_default = [
    'table of content', 'toc', 'contents',
    'bibliography', 'references',
    'list of figures', 'list of tables',
]


def filter_elements_by_title(elements: List[Element], unwanted_titles_list: List[str]) -> List[Element]:
    """
    Removes all elements that belong to a title containing the expressions of the filter strings.

    Args:
        elements (List[Element]): List of document elements.
        unwanted_titles_list (List[str]): List of expressions which occur in unwanted titles.

    Returns:
        List[Element]: Filtered list of elements.
    """
    unwanted_ids = set()
    for expression_in_title in unwanted_titles_list:
        unwanted_titles = [
            element for element in elements
            if expression_in_title.lower() in element.text.lower()
        ]
        if unwanted_titles:
            unwanted_ids.add(unwanted_titles[0].id)

    return [
        el for el in elements
        if el.metadata.parent_id not in unwanted_ids and el.id not in unwanted_ids
    ]


def filter_elements_by_unwanted_categories(elements: List[Element], unwanted_categories_list: List[str]) -> List[Element]:
    """
    Filter out elements which are contained in unwanted categories.

    Args:
        elements (List[Element]): List of document elements.
        unwanted_categories_list (List[str]): List of unwanted categories.

    Returns:
        List[Element]: Filtered list of elements.
    """
    return [
        element for element in elements
        if element.category not in unwanted_categories_list
    ]


def convert_regex_to_display(unwanted_titles_list_default):
    """
    little helper function, to convert the unwanted_titles_list_default_regex
    to displayable items. Currently not needed
    """
    display_list = []
    for regex in unwanted_titles_list_default:
        Words_or_WordGroups = regex.split('|')
        Words_or_WordGroups = [string.replace("\\s", " ").strip('^()$') for string in Words_or_WordGroups]
        display_list.append(Words_or_WordGroups)

    return [string for sublist in display_list for string in sublist]
