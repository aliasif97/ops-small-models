# Training data

Source: Supabase project "Northlane Solution - Dashboard", tables `rfq_tracker_threads` (labels) and
`rfq_tracker_messages` (text), read with SELECT only on 2026-10-12. Messages in the table span
2026-08-11 to 2026-10-13.

Rows: 1152 threads (1 thread with a null role was dropped; 1,153 in the table).

## Labels are silver labels
Every label is the output of Claude Haiku in the daily RFQ tracker. No human reviewed them.
A model trained here learns to imitate Haiku, including its mistakes. Agreement with Haiku is not accuracy.

## Fields in threads.jsonl
rn (row index), role (the label), category, company (Haiku output, kept as is), direction of the first
message (i inbound, o outbound), sender_domain of the first message, subject (first 120 chars),
text (first 250 chars of body excerpt, or the Gmail snippet when no body is stored, with email
addresses replaced by the token <email>).

## De-identification
Email addresses are not stored: the sender is reduced to its domain and addresses inside text become
<email>. Phone numbers, names inside body text, company names and subjects are kept (names and phone
numbers inside body text are not scrubbed).

## Differences from the brief
Text is the first 250 chars of the body excerpt, not 1,500. This was a cost choice for the export path
(results had to be copied through a tool channel). Longer text would probably help a little.

## Label distribution (role)
| role | count | share |
|---|---|---|
| our_rfq_to_vendor | 444 | 38.5% |
| admin_noise | 232 | 20.1% |
| customer_rfq | 196 | 17.0% |
| other | 111 | 9.6% |
| vendor_pitch | 97 | 8.4% |
| customer_followup | 56 | 4.9% |
| prequal_registration | 16 | 1.4% |

Rows containing Arabic characters: 25 (2.2%).

## Label distribution (category, top 20)
| category | count |
|---|---|
| Other | 322 |
| (none) | 164 |
| Services | 111 |
| LV/MV Electrical | 78 |
| Fire & Safety Systems | 77 |
| Cables & Wires | 67 |
| Solar & Renewables | 55 |
| Welding & Consumables | 52 |
| Lighting (non-aviation) | 45 |
| Valves & Piping | 42 |
| Transformers & Generators | 37 |
| Civil & Structures | 26 |
| Pumps & Rotating | 22 |
| Instrumentation & IT | 21 |
| Steel & Metals | 14 |
| Fasteners & Hardware | 14 |
| PPE & Safety | 5 |
