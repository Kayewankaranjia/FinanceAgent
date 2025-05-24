from langchain.prompts import PromptTemplate

class PromptTemplates:
    def __init__(self):
        self.prompt_system_task = PromptTemplate(
            template="""
                Your job is to gather a question or a query from the user about the financial metric they want to analyze.

                You should obtain the following information from them:

                - Question: The request for a metric or a goal for the analysis. e.g. "what was the percentage change in net sales from 2000 to 2001?"
                - Operation: The operation to be performed to get the metric. e.g. "percentage change, difference in, sum of, product of"
                - Timeframe: The timeframe for the analysis. If the user has entered absolute values like 2023, they mean the entire year until specified otherwise. If the user has entered a range like 2000 to 2004, you have to return all values in the range as a list. e.g. "Q1 2023, 2022, last quarter of 2024"
                - Metric: The metric or metrics that the user wants to analyze. e.g. "net sales, net income, total assets"
                return the values in a dictionary format with the keys as question, operation, timeframe, and metric.
                The question should be a string, the operation should be a string, the timeframe should be a list of strings, and the metric should be a list of strings.
            
            """
        )

        self.components_prompt = PromptTemplate(
            template="""
                For the {question}, identify the relevant components required to calculate the financial metric.
                The components include the operation to be performed, the timeframe values for the analysis, and the metric or metrics that the user wants to analyze.
                Return the values in a dictionary format with the keys as OperationToBePerformed and RequiredTimeFrames
            """,
            input_variables=["question"]
        )

        self.retrive_prompt = PromptTemplate(
            template="""
                For the given {components}, fetch the relevant values from the {data} provided.
                Usually, the values will be in data['table']. 
                The table is a matrix of values with the first column as the timeframe year or month and the rest of the columns have column names as the metric names.
                The values are usually after the first row of the table.

                Return the actual valus for the components
            """,
            input_variables=["components", "data"]
        )

        self.math_formula_prompt = PromptTemplate(
            template="""
                For the given {response}, create a formula to calculate the financial metric.
                The formula should be in the form of a string that can be evaluated using eval() function in python.
                The formula should be a simple arithmetic expression like 2 + 3 * 4
                Example:    
                Output:
                    "Percentage Change in Net Sales from 2002 to 2003" = ( <Value of Net Sales for  2003> - <Value of Net Sales for  2002>) /  <Value of Net Sales for  2003> * 100
            """,
            input_variables=["response"]
        )

        self.final_message_prompt = PromptTemplate(
            template="""
                For the asked {userquestion} identify the {answer} value and display it to the user in the correct format.  
            """,
            input_variables=["result","userquestion"]
        )

        self.file_name_prompt = PromptTemplate(    
    template="""
        For the requested {metric}, identify the relevant file name from the database.
     Use the required tools to fetch the file name from the database.
     return the values in a dictionary format with the keys as file_name
    """
)