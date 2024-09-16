import re

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


def filter_elements_by_title(elements, unwanted_titles_list):
    """
    Removes all elements that belong to a title containing the expressions of the filter strings.
    :param elements: list of elements
    :param unwanted_titles_list: list of expressions which occurs in unwanted titles
    :return:  filtered list of elements
    """
    unwanted_ids = []
    for expression_in_title in unwanted_titles_list:
        #unwanted_titles = [element for element in elements if
        #                   re.search(expression_in_title, element.text, re.IGNORECASE) and element.category == "Title"]
        unwanted_titles = [element for element in elements if
                           expression_in_title.lower() in element.text.lower()]
        if len(unwanted_titles) > 0:
            unwanted_id = unwanted_titles[0].id  # assuming there is only 1 of this specific unwanted title
            unwanted_ids.append(unwanted_id)
    filtered_elements = [el for el in elements if el.metadata.parent_id not in unwanted_ids and el.id not in unwanted_ids]
    return filtered_elements


def filter_elements_by_unwanted_categories(elements, unwanted_categories_list):
    """
    Filter out elements, which are contained in unwanted categories (Header, Footer,..)

    elements: list of unstructured.documents.elements.Element
    pdf: bool, whether the elements are from a pdf or not. Crucial for
    formula processing
    """
    # currently no multimodal RAG, filter out Image
    # UncategorizedText doesnt get filtered right now
    # for pdf documents, Formulas dont get extracted in markdown, in html
    # they are. in docx, pptx, xlsx they are ignored
    # in html, they are formatted in markdown and have no specific category
    filtered_elements = [element for element in elements if element.category not in unwanted_categories_list]
    return filtered_elements

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
