# FinanceAgent

# Create you .env file 
Use the sample values in .env.exaple 

# Create your conda env

To create and activate the conda environment. Run the following commands:

```bash
conda env create -f .\finance_bot_env.yaml
conda activate finance_bot_env
```

# Create a local Neo4j database 

Create and manage a new DBMS locally - Neo4j Desktop
https://neo4j.com/docs/desktop-manual/current/operations/create-dbms/

# Ingest data from train.json

- Navigate to src/data folder 
- Copy the train.json file 
- Execute all cells in ingest.ipynb

# Open a new terminal and type 
```
langgraph dev
```

This should open up langgraph studio. Debug with a sample question 

# Run Eval 
- Navigate to src/tests/langgraph_eval
- Execute all cells in eval.ipynb 
- This will create a new dataset and experiment in langsmith studio
