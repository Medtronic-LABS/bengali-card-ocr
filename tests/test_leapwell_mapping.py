"""Prevent loss, visit mixing and unsafe assumptions in the app review projection."""
import json
import unittest
from pathlib import Path
from pipeline.leapwell import build_app_review, destination, section_catalog
from pipeline.export import build_export_payload

ROOT = Path(__file__).resolve().parents[1]

class LeapwellMapping(unittest.TestCase):
    def test_every_paper_field_preserved_once_and_targets_exist(self):
        templates = json.loads((ROOT / 'config/layout_templates.json').read_text())
        for card, template in templates.items():
            fields = [dict(f, field_id=str(i)) for i, f in enumerate(template['cells'])]
            projection = build_app_review(card, fields)
            sources = [s for section in projection['sections'] for s in section['sources']]
            self.assertEqual(sorted(s['source_field_id'] for s in sources), sorted(f['field_id'] for f in fields))
            for section in projection['sections']:
                ids = {f['id'] for f in section['fields']}
                self.assertTrue(all(s['target_field_id'] in ids for s in section['sources']) or section['id'] == 'unmapped')

    def test_ambiguous_supplements_and_delivery_not_assumed(self):
        self.assertIsNone(destination('ANC', {'section': 'maternal_table', 'column_index': 1, 'field_key': 'iron'})['field_id'])
        self.assertIsNone(destination('ANC', {'section': 'maternal_table', 'column_index': 7, 'field_key': 'weight'})['field_id'])

    def test_epi_doses_follow_app_timeline_and_tcv_stays_empty(self):
        fields = [dict(field_id=str(c), section='table', field_key='mr', column_index=c) for c in (4, 5)]
        result = build_app_review('EPI', fields)
        sources = [s for section in result['sections'] for s in section['sources']]
        self.assertEqual([s['target_field_id'] for s in sources], ['MR1', 'MR2'])
        self.assertFalse(any(s['target_field_id'] == 'TCV' for s in sources))
        self.assertTrue(any(f['id'] == 'TCV' for s in result['sections'] for f in s['fields']))

    def test_unapproved_values_not_promoted_and_visits_separate(self):
        fields = [dict(field_id=str(i), section='table', field_key='weight', row_index=i,
                       visit_index=i, value_bn_raw=str(50+i)) for i in (1, 2)]
        result = {'card_type': 'NCD', 'fields': fields}
        payload = build_export_payload(result)
        sources = [s for section in payload['leapwell_review']['sections'] for s in section['sources']]
        self.assertEqual([s['visit_index'] for s in sources], [1, 2])
        self.assertTrue(all(s['reviewed_value'] is None for s in sources))
        self.assertFalse(payload['leapwell_review']['submission_ready'])
        reviewed = build_export_payload(result, {'1': {'review_status': 'corrected', 'edited_value': '60'}})
        sources = [s for section in reviewed['leapwell_review']['sections'] for s in section['sources']]
        self.assertEqual([s['reviewed_value'] for s in sources], ['60', None])

    def test_sections_follow_pinned_app_manifest(self):
        self.assertEqual([s['section_id'] for s in section_catalog('NCD')], ['ncdBiometrics', 'bpLog', 'symptomsLog', 'glucoseLog', 'eyeCare'])

if __name__ == '__main__':
    unittest.main()
