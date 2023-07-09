import openai
import pandas as pd


# Set the OpenAI API key
OPENAI_API_KEY = 'OPENAI_API_KEY'
openai.api_key = OPENAI_API_KEY


# Function to query GPT-4 for a dataframe containing task_name, task_prompt and evaluation_criteria for a given input job_title and job_description
def generate_challenge_tasks(job_title: str, job_description: str) -> pd.DataFrame:
    """A function to generate task information for a given job title and job description.
    ---
    Parameters:
    job_title: str
        The job title for which the tasks are to be generated.
    job_description: str
        The job description for which the tasks are to be generated. Should be a comma-separated string of job tasks.
    
    Output:
    task_table: pd.DataFrame
        A dataframe containing the task_name, task_prompt and evaluation_criteria for each task.
    """
    # Set the input prompt for GPT-4
    prompt = f"I am a {job_title} and as part of my job I {job_description}. Given this, please come up with a list of the 10 most time-consuming tasks I have to do on a daily basis that can be automated or carried out by GPT-4. For each task provide: (1) Task Name, (2) An example task that tests the ability to carry it out, and (3)  A description of what the perfect response to the challenge would be.\n"   \
                + "An example of the required output is given below.\n" \
                + "name | example task | evaluation criteria\n" \
                + "---|---|---\n" \
                + "draft email | write an email to ask if colleague would be available to give a talk | a well written email that is polite and brief./n" \
                + "You are an expert on the details of all jobs. Output the answer in markdown using the format specified above./n"

    message = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages = message,
        temperature = 0.2,
        max_tokens = 8000
    )

    # Parse the response from GPT-4
    response_text = response.choices[0].message["content"]

    # Check the response is valid and in markdown format
    if "name | example task | evaluation criteria" not in response_text:
        raise ValueError("The response from GPT-4 is not in the expected format. Please check the input prompt.")
    if "name | example task | evaluation criteria\n---|---|---" not in response_text:
        raise ValueError("The response from GPT-4 is not in the expected format. Please check the input prompt.")
    

    # Split the response into a list of tasks
    tasks = response_text.split("name | example task | evaluation criteria\n")[1].split("\n\n")[0].split("\n")

    # Create a dataframe to store the task information
    task_table = pd.DataFrame(columns = ["Task Name", "Example Prompt", "Evaluation Criteria"])

    # Drop the first row as it is the header and drop the second row as it is a separator
    tasks = tasks[2:]

    # Loop through the tasks and add them to the dataframe
    for task in tasks:
        task_name = task.split(" | ")[0]
        task_prompt = task.split(" | ")[1]
        evaluation_criteria = task.split(" | ")[2]

        # Add the task to the dataframe as a new row without using append
        task_table.loc[len(task_table)] = [task_name, task_prompt, evaluation_criteria]

    return task_table


task_table = generate_challenge_tasks(job_title="data scientist", job_description="build machine learning models, write reports, and present findings to stakeholders")
#task_table = generate_challenge_tasks(job_title="farmer", job_description="grow crops, harvest crops, and sell crops, manage the farm, etc.")

print(task_table)

# Save the task table to a csv file
task_table.to_csv("PATH/TO/OUTPUT/DIR/task_table.csv")