import os
from google.protobuf.json_format import MessageToJson
from typing import List
from instabase.protos.ibprog import ibrefiner_prog_pb2

from instabase.training_utils.model_artifact import ModelArtifactContext

_MIN_CONFIDENCE_FIELD_NAME = "__MIN_CONFIDENCE_VALIDATION"
_MIN_CONFIDENCE_DEFAULT_VALUE = "0.90"

_PUBLISHED_MODEL_COL_NAME = "__PUBLISHED_MODEL_VERSION"
_PUBLISHED_MODEL_VALUE = "'{{published_model_version}}'"
_PUBLISHED_MODEL_NAME = "__PUBLISHED_MODEL_NAME"


def _make_refiner_field(label: str, function: str, output_type: str = "") -> ibrefiner_prog_pb2.RefinerField:
    return ibrefiner_prog_pb2.RefinerField(
        desc=ibrefiner_prog_pb2.RefinerFieldDescription(
            output_type=output_type,
            label=label,
        ),
        req=ibrefiner_prog_pb2.RefinerRequest(expression=function, type=ibrefiner_prog_pb2.REFINER_FUNC),
    )


def _make_run_model_script_contents() -> str:
    return f"""
import json
from typing import Any, Dict, Union, List, cast
from ib.market.ib_intelligence.functions import IntelligencePlatform, kwargs_to_ms_params, resolve_input, log
from instabase.ocr.client.libs.ibocr import ParsedIBOCRBuilder
from instabase.provenance.tracking import Value

# we expect this format to come out ner_result_to_values(), and is used as input for get_confidence() and get_field()
FieldType = Value[Any]
ModelResultType = Value[Dict[str, Value[List[FieldType]]]]
ModelResultNoProvType = Dict[str, List[List[Union[str, float]]]]

def table_result_to_values(result: Any, INPUT_COL: Value[str]) -> ModelResultType:
    raw_data: bytes = result.get('raw_data', {{}}).get('data', [])
    json_result = json.loads(raw_data.decode("utf-8"))
    fields = json_result['fields']
    results: Dict[str, Any] = {{}}

    for field in fields:
        table_annotations = field['table_annotations']
        results[field['field_name']] = []
        for table_annotation in table_annotations:
            row_len, col_len = len(table_annotation['rows']), len(table_annotation['cols'])
            cells = [['' for _ in range(col_len)] for i in range(row_len)]
            for cell in table_annotation['cells']:
                # TODO(qxie3): handle merge cell
                # we currently assume start_index == end_index before supporting merge cell
                cells[cell['row_start_index']][cell['col_start_index']] = INPUT_COL[cell['start_index']:cell['end_index']]
            results[field['field_name']].append(Value(cells))
        results[field['field_name']] = Value(results[field['field_name']])

    return Value(results)


def ner_result_to_values(result: Any, INPUT_COL: Value[str]) -> ModelResultType:
  entities = result.get('ner_result', {{}}).get('entities', [])

  fields: Dict[str, Value[List[Any]]] = {{}}

  for ent in entities:
    if ent['label'] == 'O':
      continue
    if ent['label'] not in fields:
      fields[ent['label']] = Value([])

    fields[ent['label']].value().append(
      Value([
        INPUT_COL[ent['start_index']:ent['end_index']],
        ent['score']
      ])
    )

  return Value(fields)


def average(values: List[float], **kwargs) -> Union[float, str]:
  if len(values) == 0:
    return 'undefined'
  return sum(values) / len(values)

def get_field(MODEL_RESULT_COL: ModelResultType, field_name: Value[str], **kwargs) -> Value[str]:
  \"\"\"Returns space-joined values of an extracted field

  Equivalent to the following:
    join(' ',
      map(
        map_get(
          MODEL_RESULT_COL,
          field_name,
          default=list()
        ),
        'first(x)'
      )
    )

  Args:
    MODEL_RESULT_COL(dict): dictionary containing model output
    field_name(str): field name

  Returns:
    Returns the extracted values, joined together by whitespace.

  \"\"\"
  if 'result_type' in MODEL_RESULT_COL.value() and MODEL_RESULT_COL.value().get('result_type') == 'Table':
    field_values =  MODEL_RESULT_COL.value().get(field_name.value(), Value([])) 
    return Value(field_values.value()[0])
  else:
    field_values =  MODEL_RESULT_COL.value().get(field_name.value(), Value([]))
    vals = [x.value()[0] for x in field_values.value()]  
    return Value.join(' ', vals)

# type for error messages
ErrorStr = str

def get_confidence_no_provenance(MODEL_RESULT_COL: ModelResultNoProvType, field_name: str, **kwargs) -> Union[float, ErrorStr]:
  \"\"\"Returns the confidence score of an extracted field

  Equivalent to the following:
    average(
      map(
        map_get(
          MODEL_RESULT_COL,
          field_name,
          default=list()
        ),
        'last(x)'
      )
    )

  Args:
    MODEL_RESULT_COL(dict): dictionary containing model output
    field_name(str): field name

  Returns:
    Returns the confidence score, between 0 and 1.

  \"\"\"
  field_values = MODEL_RESULT_COL.get(field_name, [])
  confidences = [ cast(float, conf) for val, conf in field_values ]
  if len(confidences) == 0:
    return 'Error: no confidences obtained'

  return average(confidences)

def get_field_no_provenance(MODEL_RESULT_COL: ModelResultNoProvType, field_name: str, **kwargs) -> str:
  \"\"\"Returns space-joined values of an extracted field

  Equivalent to the following:
    join(' ',
      map(
        map_get(
          MODEL_RESULT_COL,
          field_name,
          default=list()
        ),
        'first(x)'
      )
    )

  Args:
    MODEL_RESULT_COL(dict): dictionary containing model output
    field_name(str): field name

  Returns:
    Returns the extracted values, joined together by whitespace.

  \"\"\"
  raise NotImplementedError("Provenance Tracking must be on to call get_field. Please turn it on in File > Settings.")


def run_model_no_provenance(INPUT_COL: str, model_version: str, **kwargs) -> Any:
  \"\"\"
  Runs the trained model associated with this refiner program.
  \"\"\"
  raise NotImplementedError("Provenance Tracking must be on to run a model. Please turn it on in File > Settings.")

def run_model(INPUT_COL: Value[str], model_name: Value[str], model_version: Value[str], **kwargs) -> ModelResultType:
  ip_sdk = IntelligencePlatform(kwargs)
  ibocr, err = kwargs['_FN_CONTEXT_KEY'].get_by_col_name('INPUT_IBOCR_RECORD')
  if err:
    raise KeyError(err)
  record = ParsedIBOCRBuilder(use_ibdoc=True)
  record.add_ibocr_records([ibocr])
  results = ip_sdk.run_model(model_name.value(),
                          input_record=record, 
                          force_reload=False,
                          refresh_registry=False,
                          model_version=model_version.value(),
                          **kwargs)
  if 'ner_result' in results:
      return ner_result_to_values(results, INPUT_COL)
  elif 'raw_data' in results and results['raw_data'].get('type') == 'Table':
      return table_result_to_values(results, INPUT_COL)
  else:
      raise TypeError('run_model returns unexpected result type.')

def register(name_to_fn):
  name_to_fn.update({{
    'run_model': {{
      'fn': run_model_no_provenance, # Provenance tracking is required
      'fn_v': run_model,
      'ex': '',
      'desc': '',
    }},
    'get_confidence': {{
      'fn': get_confidence_no_provenance,
      'ex': '',
      'desc': '',
    }},
    'get_field': {{
      'fn': get_field_no_provenance,
      'fn_v': get_field,
      'ex': '',
      'desc': '',
    }},
    'average': {{
      'fn': average,
      'ex': '',
      'desc': '',
    }}
  }})
  """


def write_refiner_program(
    context: ModelArtifactContext,
    ib_model_path: str,
    labels: List[str],
    model_name: str,
    dev_input_path: str,
    is_table: bool = False,
) -> str:
    """Writes a refiner module to the training job output, which can be imported
    into a Flow. Returns the local path to the Refiner program.
    """

    refiner_module_path = os.path.join(context.tmp_dir.name, "refiner_mod")
    refiner_prog_folder = os.path.join(refiner_module_path, "prog")
    refiner_path = os.path.join(refiner_prog_folder, f"{model_name}_refiner.ibrefiner")
    refiner_scripts_path = os.path.join(refiner_module_path, "scripts")

    os.makedirs(refiner_module_path, exist_ok=True)
    os.makedirs(refiner_scripts_path, exist_ok=True)
    os.makedirs(refiner_prog_folder, exist_ok=True)

    # write script that can load model
    script_path = os.path.join(refiner_scripts_path, "run_model.py")
    script_content = _make_run_model_script_contents()
    with open(script_path, "w+") as f:
        f.write(script_content)

    ibrefiner = ibrefiner_prog_pb2.IBRefinerProg(
        options=ibrefiner_prog_pb2.RefinerProgOptions(
            provenance_tracking=True,
            auto_provenance=False,
            scripts_path="../scripts",
        ),
        dev_input=ibrefiner_prog_pb2.DevelopmentInput(
            input_path=dev_input_path,
            input_files=None,
        ),
    )

    # create constant field for minimum confidence
    min_confidence_field = _make_refiner_field(_MIN_CONFIDENCE_FIELD_NAME, _MIN_CONFIDENCE_DEFAULT_VALUE)
    min_confidence_field.desc.description = (
        "Set a confidence between 0-1 that all fields must exceed to pass validation. Set to 0 for no validation."
    )
    ibrefiner.fields.append(min_confidence_field)

    # Create constant for the published model version to use
    model_version_field = _make_refiner_field(_PUBLISHED_MODEL_COL_NAME, _PUBLISHED_MODEL_VALUE)
    model_version_field.desc.description = "Set the model version published in Marketplace to use in this Refiner."
    ibrefiner.fields.append(model_version_field)

    model_name_field = _make_refiner_field(_PUBLISHED_MODEL_NAME, f"echo('{model_name}')")
    model_name_field.desc.description = "Set the model name published in Marketplace to use in this Refiner."
    ibrefiner.fields.append(model_name_field)

    # create hidden field for model result
    ibrefiner.fields.append(
        _make_refiner_field(
            "__model_result", f"run_model(INPUT_COL, {_PUBLISHED_MODEL_NAME}, {_PUBLISHED_MODEL_COL_NAME})"
        )
    )

    # create fields for each of extracted_fields
    for label in labels:
        label_no_spaces = label.replace(" ", "_")
        # create main field
        if is_table:
            ibrefiner.fields.append(
                _make_refiner_field(label_no_spaces, f"get_field(__model_result, '{label}')", "Table")
            )
        else:
            ibrefiner.fields.append(_make_refiner_field(label_no_spaces, f"get_field(__model_result, '{label}')"))

        # create confidence field
        ibrefiner.fields.append(
            _make_refiner_field(f"__{label_no_spaces}_confidence", f"get_confidence(__model_result, '{label}')")
        )

        # create validation field
        # ibrefiner.fields.append(
        #     _make_refiner_field(
        #         f'__validate_{label_no_spaces}',
        #         f"validate(['assert_true(__{label_no_spaces}_confidence > {_MIN_CONFIDENCE_FIELD_NAME})'])"
        #     ))

    # write refiner program
    with open(refiner_path, "w+") as f:
        f.write(MessageToJson(ibrefiner, use_integers_for_enums=False))

    return refiner_module_path
