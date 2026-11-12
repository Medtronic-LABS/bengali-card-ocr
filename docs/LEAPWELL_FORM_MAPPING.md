# LEAPWELL form mapping

The reference is [Medtronic-LABS/uhis_lf_mobile](https://github.com/Medtronic-LABS/uhis_lf_mobile/tree/26380a20cdbaa2224ba6f1619ae92f62c8417139), the LEAPWELL Flutter app shipped on Android. Its release plan and pubspec identify LEAPWELL explicitly.

The review app groups fields by the app's form sections and includes an all-fields overview. JSON exports add `leapwell_review` with app destinations, original source field IDs, separate visits, raw candidates and explicitly reviewed values. The original source-field export is retained. No values are sent to the app.

## Source of truth

Pinned copies of `assets/forms/layout_manifests.json`, `field_library.json` and `epi_schedule.json` live in `config/leapwell/`. `source.json` records the repository revision and source blob hashes. Runtime uses these local files, with no network dependency. Manifest order and field IDs are preserved. Blank manifest section titles use the field-library label or a readable section ID.

- ANC: medical history, specific vitals, laboratory investigations, vaccination/supplements, ANC services/birth preparedness, danger signs.
- NCD: biometrics, BP/vitals, symptoms, blood glucose, eye care.
- EPI: birth, 6/10/14 weeks, 9/15 months, with the app's vaccine codes and display names. TCV exists in the app and stays unfilled because these paper templates have no TCV source.
- ANC card review also lists the app's pregnancy outcome, mother PNC and child sections, without treating the paper's three delivery columns as three PNC visits.

## Mapping boundaries

This is a source-linked review projection, not an API-ready payload or a full mobile UI reproduction. App options and widget metadata accompany the field catalog; enum conversion, date conversion, unit validation, patient eligibility, conditional visibility and combined-programme deduplication remain the app's responsibility and are not reproduced here. `submission_ready` is always false.

Every paper field appears exactly once, either under a known app review destination or under “Paper details — mapping needed”. Paper visit indices remain separate. Name/age/household context is retained in that latter group pending patient matching. Unknown units, supplement consumed-versus-provided amounts, TT completion, free-text referrals, newborn details and delivery columns are not silently converted into app answers. A blank vaccine cell never becomes a Missed record. App-only fields remain visibly unfilled; this view is not an assessment-completion claim.

`tests/test_leapwell_mapping.py` covers complete source retention, valid destination IDs, visit separation, review approval, EPI dose placement and ambiguous mapping. Nothing has been installed on Android or submitted to the app's backend.
