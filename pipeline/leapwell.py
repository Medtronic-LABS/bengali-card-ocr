"""Pinned LEAPWELL form structure for OCR review, not an API payload mapper.

Preserves individual paper visits. Target IDs identify review destinations only;
raw values are never coerced into app enums, dates, units or clinical decisions.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent / 'config' / 'leapwell'


@lru_cache(maxsize=1)
def catalog():
    read = lambda name: json.loads((CONFIG / name).read_text())
    return read('layout_manifests.json'), read('field_library.json'), read('epi_schedule.json'), read('source.json')


def readable(key):
    return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', key).capitalize()


def section_catalog(card_type):
    manifests, library, schedule, _ = catalog()
    forms = {'ANC': ['anc', 'pregnancyOutcome', 'pncMother', 'pncChild'], 'NCD': ['ncd'], 'EPI': []}[card_type]
    sections = []
    for form in forms:
        manifest = next(m for m in manifests if m['formType'] == form)
        for section in manifest['sections']:
            sid = section['sectionId']
            fields = []
            for ref in section['fieldRefs']:
                ref = {'id': ref} if isinstance(ref, str) else ref
                fid = ref['id']
                definition = library[fid]
                fields.append({'id': fid, 'label': ref.get('fieldName') or definition.get('label') or readable(fid),
                               'label_bn': ref.get('fieldNameCulture') or definition.get('titleCulture') or '',
                               'widget': definition.get('widgetHint'), 'options': definition.get('optionsList', [])})
            sections.append({'id': form + '/' + sid, 'form_type': form, 'section_id': sid,
                             'label': section.get('title') or library.get(sid, {}).get('label') or readable(sid), 'fields': fields})
    for milestone in schedule if card_type == 'EPI' else []:
        sections.append({'id': 'epi/' + milestone['milestoneKey'], 'form_type': 'CHILD_IMMUNIZATION',
                         'section_id': milestone['milestoneKey'], 'label': milestone['milestone'],
                         'fields': [{'id': v['code'], 'label': v['display'], 'label_bn': '', 'widget': 'vaccination_record'} for v in milestone['vaccines']]})
    return sections


# Only unambiguous paper-to-app concepts. An app ID is a review destination,
# not proof of compatible enum/unit/value. Ambiguous concepts stay unmapped.
ANC = {'weight': ('ancSpecificVitals', 'weight'), 'height': ('ancSpecificVitals', 'height'),
       'temperature': ('ancSpecificVitals', 'temperature'), 'pulse': ('ancSpecificVitals', 'pulse'),
       'blood_pressure': ('ancSpecificVitals', 'bloodPressure'), 'edema': ('ancSpecificVitals', 'edema'),
       'fundal_height': ('ancSpecificVitals', 'fundalHeight'), 'fetal_movement': ('ancSpecificVitals', 'fetalMovement'),
       'hemoglobin': ('labInvestigations', 'hemoglobin'), 'urine_albumin': ('labInvestigations', 'urinaryAlbumin'),
       'urine_bilirubin': ('labInvestigations', 'urinaryBilirubin'), 'glucose': ('labInvestigations', 'glucoseType'),
       'usg': ('ancServicesBirthPreparedness', 'ultrasound')}
NCD = {'height': ('ncdBiometrics', 'height'), 'weight': ('ncdBiometrics', 'weight'),
       'blood_pressure': ('bpLog', 'bpLogDetails'), 'glucose_fasting': ('glucoseLog', 'glucose'),
       'glucose_random': ('glucoseLog', 'glucose'), 'eye_exam': ('eyeCare', 'eyeTestOutcome')}


def destination(card_type, field):
    key = field['field_key']
    target = None
    note = 'Review destination only; app value format and units still require validation.'
    if card_type == 'ANC' and field.get('section') == 'maternal_table' and field.get('column_index', 0) <= 6:
        if key in ANC:
            section, target = ANC[key]
            section = 'anc/' + section
    elif card_type == 'NCD' and field.get('section') == 'table' and key in NCD:
        section, target = NCD[key]
        section = 'ncd/' + section
    elif card_type == 'EPI' and field.get('section') == 'table':
        col = field.get('column_index')
        dose = {'bcg': {1: 'BCG'}, 'ipv': {1: 'FIPV1', 3: 'FIPV2'}, 'mr': {4: 'MR1', 5: 'MR2'}}.get(key, {}).get(col)
        if key in ('penta', 'pcv', 'opv') and col in (1, 2, 3):
            dose = key.upper() + str(col)
        if dose:
            section = next(s['id'] for s in section_catalog('EPI') if any(f['id'] == dose for f in s['fields']))
            target = dose
            note = 'Verify vaccine dose and full date. Blank does not mean missed; vaccination status is not inferred.'
    if target is None:
        return {'section': 'unmapped', 'field_id': None, 'status': 'mapping_required',
                'note': 'No unambiguous app destination. Keep the paper value for manual mapping; delivery columns are not assumed to be PNC visits.'}
    return {'section': section, 'field_id': target, 'status': 'candidate_destination', 'note': note}


def build_app_review(card_type, fields):
    """Project raw or exported fields into app sections without dropping any source."""
    sections = section_catalog(card_type)
    for section in sections:
        section['sources'] = []
    sections.append({'id': 'unmapped', 'form_type': None, 'section_id': None,
                     'label': 'Paper details — mapping needed', 'fields': [], 'sources': []})
    index = {s['id']: s for s in sections}
    for field in fields:
        dest = destination(card_type, field)
        index[dest['section']]['sources'].append({
            'source_field_id': field['field_id'], 'source_key': field['field_key'],
            'target_field_id': dest['field_id'], 'mapping_status': dest['status'], 'mapping_note': dest['note'],
            'visit_index': field.get('visit_index'), 'paper_row': field.get('row_index'),
            'paper_column': field.get('column_index'), 'delivery_column_index': field.get('delivery_column_index'),
            'candidate': field.get('value_bn_raw', field.get('model_prediction', {}).get('value_bn_raw')),
            'reviewed_value': field.get('final_value'),
            'glucose_type_hint': {'glucose_fasting': 'FBS', 'glucose_random': 'RBS'}.get(field['field_key']),
        })
    return {'source': catalog()[3], 'submission_ready': False,
            'note': 'App section review projection. Missing fields are not inferred. Patient-dependent visibility and API serialization are not implemented.',
            'sections': sections}
