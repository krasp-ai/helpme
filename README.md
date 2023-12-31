> **Warning**
> The *Helpme* project is at a very early stage. Use with caution (if there is anything to use).


# helpme
Understand (or grasp) how LLMs can help you for your job.

## Abstract

The integration of large language models (LLMs) in your work can be challenging, especially if none of your peers use LLMs. Looking at conventional LLM benchmarks provides you with very little useful information, as likely none of the tasks you do on a daily basis are directly assessed. In this project, we propose a novel way of creating custom benchmarks that evaluate how LLMs can help you in your specific job. Whilst these benchmarks cannot provide exhaustive proof, if LLMs can help you well on an explicit task, they can provide a first starting point for making use of the capabilities of LLMs in your work. Our approach uses LLMs to automatically generate the custom benchmarks for each user's speific requirements. We share an open-source implementation and provide a publicly accessible user interface to try out our method.

## Project outputs

- GitHub repo
- Python package
- Paper
- User App

## Example usage

#### 1. User input: job title and description (in **bold**)

  Prompt to LLM:

  > I am a **researcher** and as part of my job I **read papers, write emails to colleagues, apply for grants**. Please come up with a list of 10 tasks I have to do on a daily basis, and give an example challenge for each task that tests the ability to do this task. An example is given below.
  >
  > ```
  > name | challenge | evaluation criteria
  > ---|---|---
  > draft email | write an email to ask if colleague would be available to give a talk | a well written email that is polite and brief
  > ```
  >
  > Format your answer in the same markdown format.

#### 2. Output (actual output from GPT-4):

  name | challenge | evaluation criteria
  ---|---|---
  Literature Review | Identify and summarize the key points of five relevant papers in your research field | A well-written summary with clear identification of major findings and relevance to own research
  Draft Email | Write an email to ask if colleague would be available to give a talk | A well-written email that is polite, brief, and effective
  Grant Writing | Draft a one-page abstract for a potential research grant application | A concise and compelling abstract that effectively communicates the research question, methodology, and potential impact
  Data Analysis | Analyze a given dataset and write a brief report on your findings | A clear and comprehensive analysis report with accurate conclusions and appropriate use of statistical tools
  Lab Management | Develop a schedule for equipment usage in the lab for a week | An efficient and fair schedule that accounts for all necessary tasks and lab members
  Manuscript Writing | Write the introduction for a new research paper | A well-written introduction that clearly states the research question, significance, and brief overview of the approach
  Mentorship | Provide feedback on a junior researcher's report | Constructive and detailed feedback that helps improve the quality of the report
  Conference Preparation | Create a 15-minute presentation on a recent study for an upcoming conference | A clear, engaging, and informative presentation that covers all major aspects of the study
  Collaboration | Develop a plan for a new collaborative research project with another lab | A detailed and feasible project plan that includes clear roles, milestones, and potential challenges
  Peer Review | Review a submitted paper in your field and provide a critique | A fair and thorough review that assesses both the strengths and weaknesses of the paper

#### 3. Visualisation

Using [Krasp.ai](https://krasp.ai/) we can then run and visualise these challenges so that the user has a better idea where LLMs can help them in their job – and where they fail to help.

## User interface mock-up

![userinterface_demo](https://github.com/krasp-ai/helpme/assets/75615911/82efd45f-27d2-4eae-bc6e-9cfa9bd4fb1d)


## Tasks

- Literature review: where do we want to position our paper? In the automatic LLM evaluation space, interpretability, or ...?
- Langchain generation code: given a job title and description, create a list of tasks and corresponding challenges?
- User interface: built into Krasp.ai or via Gradio
- Create examples of good benchmarks: define what a useful benchmark would be.

## License

All of the code in this repository is licensed under [Apache license v2.0](https://github.com/krasp-ai/helpme/blob/main/LICENSE), copyright 2023 Punaltz Ltd.
