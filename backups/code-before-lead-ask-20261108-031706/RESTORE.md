# Code backup before the "lead diagnosis -> investigate -> ask the customer" change (20261108-031706)

Restores the agent code exactly as it was before the change. Data (app.db, checkpoints, memory, Qdrant) is not
part of this backup and is not touched by restoring it.

From the project root (D:\Agent_Latest), with the API server stopped:

    cp -r backups/code-before-lead-ask-20261108-031706/backend/app      backend/
    cp -r backups/code-before-lead-ask-20261108-031706/backend/evals    backend/
    cp -r backups/code-before-lead-ask-20261108-031706/backend/scripts  backend/
    cp -r backups/code-before-lead-ask-20261108-031706/frontend         .
    cp    backups/code-before-lead-ask-20261108-031706/README.md        README.md

Then restart the API server.
