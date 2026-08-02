# TRI-NETRA AGENT RULES

======================================================================
PERMANENT TEMPORARY-ARTIFACT POLICY
======================================================================

This rule applies to ALL current and future TRI-NETRA development work.

During implementation, testing, validation, debugging, benchmarking, or
auditing, you MAY create temporary development artifacts when genuinely
necessary.

Examples include:

- temporary test files;
- pytest test files;
- validation scripts;
- audit scripts;
- generated JSON reports;
- generated CSV reports;
- benchmark outputs;
- inspection scripts;
- debug files;
- temporary Markdown reports;
- completion reports;
- freeze reports;
- intermediate analysis outputs;
- temporary logs;
- temporary statistics files.

However, these artifacts are NOT automatically considered permanent
repository files.

----------------------------------------------------------------------
MANDATORY LIFECYCLE
----------------------------------------------------------------------

Whenever you create any temporary development artifact:

1. CREATE it only if it is actually useful for the current task.

2. USE it to perform the required testing, validation, debugging,
   benchmarking, or analysis.

3. SHOW the user the important results.

4. Clearly identify which files were created temporarily.

5. DO NOT delete them yet.

6. Ask the user for permission to clean them up.

Use a message similar to:

"Validation is complete.

Temporary artifacts created during this stage:

- tests/test_stage5_fusion.py
- scripts/validate_stage5.py
- stage5_validation.json

These files were used only for testing/validation and are not required
by the runtime implementation.

Should I delete these temporary artifacts now?"

7. WAIT FOR EXPLICIT USER APPROVAL.

8. ONLY AFTER the user approves deletion, remove the temporary
   artifacts.

9. After deletion, remove empty directories and generated cache files
   where appropriate.

----------------------------------------------------------------------
CRITICAL DELETION RULE
----------------------------------------------------------------------

NEVER automatically delete a file that you created during the current
task without first asking the user.

Even if the file is obviously temporary.

The sequence must always be:

CREATE
   ↓
USE
   ↓
VALIDATE
   ↓
SHOW RESULTS
   ↓
LIST TEMPORARY FILES
   ↓
ASK USER
   ↓
USER APPROVES
   ↓
DELETE


NOT:

CREATE
   ↓
USE
   ↓
DELETE AUTOMATICALLY


The user must have an opportunity to inspect the results before cleanup.

----------------------------------------------------------------------
WHAT COUNTS AS TEMPORARY
----------------------------------------------------------------------

Examples:

tests/test_stage5_*.py
tests/test_stage6_*.py

scripts/validate_stage5.py
scripts/audit_stage5.py

stage5_report.json
stage5_metrics.json
validation_results.json

Stage_5_Completion_Report.md
Stage_5_Freeze_Report.md

inspection.py
debug.py
benchmark.py

temporary CSV/JSON outputs

and similar development-only artifacts.

These should be treated as candidates for cleanup after validation.

----------------------------------------------------------------------
WHAT IS NOT TEMPORARY
----------------------------------------------------------------------

Do NOT classify actual implementation files as temporary.

Examples:

src/canonical/*
src/models/*
src/resolution/*
src/correlation/*
src/fusion/*
future feature-engineering modules
future risk-engine modules
future API/backend modules

Also protect:

data/*
README.md
requirements.txt
LICENSE
.gitignore

and:

docs/TRI_NETRA_STAGE_WISE_DOCUMENTATION.md


These must not be deleted under the temporary-artifact cleanup policy.

----------------------------------------------------------------------
DOCUMENTATION POLICY
----------------------------------------------------------------------

The preferred permanent Markdown files are:

README.md

and:

docs/TRI_NETRA_STAGE_WISE_DOCUMENTATION.md


If you create temporary:

Completion_Report.md
Freeze_Report.md
Audit_Report.md
Testing_Report.md
Benchmark_Report.md

or similar Markdown documents, show their results and then ask whether
the user wants them deleted.

Do NOT assume they should remain permanently.

----------------------------------------------------------------------
TESTING POLICY
----------------------------------------------------------------------

Tests MAY be created freely when required to verify correctness.

Do not avoid testing merely because tests may later be deleted.

Correct workflow:

implementation
→ temporary tests
→ execute tests
→ fix failures
→ execute again
→ report final result
→ ask user whether temporary tests should be deleted


Never sacrifice validation quality merely to keep the repository clean.

----------------------------------------------------------------------
JSON / GENERATED OUTPUT POLICY
----------------------------------------------------------------------

Generated JSON files should normally be considered temporary unless they
are explicitly part of the application's runtime architecture.

For example:

stage5_report.json
stage5_validation.json
benchmark_results.json

should normally follow:

generate
→ use
→ show result
→ ask permission
→ delete if approved


But an actual runtime configuration/data contract JSON must NOT be
deleted simply because its extension is .json.

Always distinguish:

GENERATED DEVELOPMENT ARTIFACT

from:

APPLICATION FILE.

----------------------------------------------------------------------
CACHE FILES
----------------------------------------------------------------------

Generated caches such as:

__pycache__/
*.pyc
.pytest_cache/

may be cleaned during repository cleanup.

However, if performing a broader cleanup involving other temporary
files, still clearly state what is being removed.

----------------------------------------------------------------------
FINAL RULE
----------------------------------------------------------------------

TRI-NETRA follows:

BUILD
→ TEST
→ VALIDATE
→ SHOW
→ ASK
→ CLEAN
→ PROCEED


Never:

BUILD
→ TEST
→ SILENTLY DELETE EVIDENCE


Repository cleanliness is important, but user approval is required
before deleting temporary development artifacts.

======================================================================
GIT POLICY
======================================================================
- NEVER push to git
- NEVER commit anything using git
- ALL git operations are handled manually by the user on their side.
