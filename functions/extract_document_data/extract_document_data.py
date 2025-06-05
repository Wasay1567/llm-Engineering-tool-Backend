import os
from functions.extract_document_data.parse_docx import parse_docx
from functions.extract_document_data.parse_pdf import parse_pdf
from functions.extract_document_data.parse_txt_file import parse_txt_file
from functions.extract_image_data.extract_image_data import extract_image_data
import io

async def extract_document_data(document):
    """
    Extracts and processes data from a given document based on its content type.

    This asynchronous function reads the provided document, determines its
    content type, and processes it accordingly to extract its data. The supported
    document types are PDF, DOCX, plain text.

    :param document: The file-like object representing the document to process.
    :return: The processed data extracted from the document as text.
    :rtype: str
    """

    file_data = await document.read()
    content_type = document.content_type
    _, file_extension = os.path.splitext(document.filename)

    if content_type == 'application/pdf' or file_extension == '.pdf':
        return parse_pdf(file_data)
    elif content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' or file_extension == '.docx':
        return parse_docx(file_data)
    elif content_type == "text/plain" or file_extension == '.txt':
        return parse_txt_file(file_data)
    else:
        return None
