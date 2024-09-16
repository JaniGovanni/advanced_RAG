from unstructured_client import UnstructuredClient
from unstructured_client.models import operations, shared
from unstructured_client.models.errors import SDKError
from unstructured.staging.base import dict_to_elements
import os


def process_api(input_filepath):
    client = UnstructuredClient(
        api_key_auth=os.getenv("UNSTRUCTURED_API_KEY"),
        server_url=os.getenv("UNSTRUCTURED_API_URL"),
    )
    with open(input_filepath, "rb") as f:
        files = shared.Files(content=f.read(), file_name=input_filepath)

    req = shared.PartitionParameters(
        files=files,
        strategy="hi_res",
        hi_res_model_name="yolox",
        pdf_infer_table_structure=True,
        skip_infer_table_types=[],
    )

    try:
        resp = client.general.partition(req)
        return dict_to_elements(resp.elements)
    except SDKError as e:
        print(e)
        return None

