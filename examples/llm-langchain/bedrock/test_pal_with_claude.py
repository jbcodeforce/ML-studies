import TestBedrockCoT

def buildPrompt():
    DATE_UNDERSTANDING_PROMPT = """
    # Q: 2015 is coming in 36 hours. What is the date one week from today in MM/DD/YYYY?
    # If 2015 is coming in 36 hours, then today is 36 hours before.
    today = datetime(2015, 1, 1) - relativedelta(hours=36)
    # One week from today,
    one_week_from_today = today + relativedelta(weeks=1)
    # The answer formatted with %m/%d/%Y is
    one_week_from_today.strftime('%m/%d/%Y')
    # Q: The first day of 2019 is a Tuesday, and today is the first Monday of 2019. What is the date today in MM/DD/YYYY?
    # If the first day of 2019 is a Tuesday, and today is the first Monday of 2019, then today is 6 days later.
    today = datetime(2019, 1, 1) + relativedelta(days=6)
    # The answer formatted with %m/%d/%Y is
    today.strftime('%m/%d/%Y')
    # Q: The concert was scheduled to be on 06/01/1943, but was delayed by one day to today. What is the date 10 days ago in MM/DD/YYYY?
    # If the concert was scheduled to be on 06/01/1943, but was delayed by one day to today, then today is one day later.
    today = datetime(1943, 6, 1) + relativedelta(days=1)
    # 10 days ago,
    ten_days_ago = today - relativedelta(days=10)
    # The answer formatted with %m/%d/%Y is
    ten_days_ago.strftime('%m/%d/%Y')
    # Q: It is 4/19/1969 today. What is the date 24 hours later in MM/DD/YYYY?
    # It is 4/19/1969 today.
    today = datetime(1969, 4, 19)
    # 24 hours later,
    later = today + relativedelta(hours=24)
    # The answer formatted with %m/%d/%Y is
    today.strftime('%m/%d/%Y')
    # Q: Jane thought today is 3/11/2002, but today is in fact Mar 12, which is 1 day later. What is the date 24 hours later in MM/DD/YYYY?
    # If Jane thought today is 3/11/2002, but today is in fact Mar 12, then today is 3/12/2002.
    today = datetime(2002, 3, 12)
    # 24 hours later,
    later = today + relativedelta(hours=24)
    # The answer formatted with %m/%d/%Y is
    later.strftime('%m/%d/%Y')
    # Q: Jane was born on the last day of Feburary in 2001. Today is her 16-year-old birthday. What is the date yesterday in MM/DD/YYYY?
    # If Jane was born on the last day of Feburary in 2001 and today is her 16-year-old birthday, then today is 16 years later.
    today = datetime(2001, 2, 28) + relativedelta(years=16)
    # Yesterday,
    yesterday = today - relativedelta(days=1)
    # The answer formatted with %m/%d/%Y is
    yesterday.strftime('%m/%d/%Y')
    # Q: {question}
    """.strip() + '\n'
    return DATE_UNDERSTANDING_PROMPT

def extractPythonCodeFromResponse(response):
    result = ""
    for line in response.split("\n"):
        if line.startswith("```") and not line.startswith("```python"):
            return result
        if not line.startswith("```python") and not "* " in line:
            result+=line+"\n"
    return result

def main():
    MODEL_NAME="anthropic.claude-v2"
    prompt=buildPrompt()

    question = "Today is 27 February 2023. I was born exactly 25 years ago. What is the date I was born in MM/DD/YYYY?"
    prompt = prompt.format(question= question)
    print(prompt)

    print("\n--- Try with standard Claude parameter")
    claude_inference_modifier = {
        "max_tokens_to_sample": 1024,
        "temperature": 0.5,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": ["\n\nHuman"],
    }
    generatedCode=TestBedrockCoT.textPlayground(MODEL_NAME,prompt,claude_inference_modifier)
    print(generatedCode)

    print("\n\n--- Try with temperature =0 ... it generates python code")
    claude_inference_modifier["temperature"]=0
    response=TestBedrockCoT.textPlayground(MODEL_NAME,prompt,claude_inference_modifier)

if __name__ == '__main__':
    response="""
* Today is 27 February 2023
* I was born exactly 25 years ago
* So my birthday is 25 years before 27 February 2023
* 25 years before 27 February 2023 is 27 February 1998
* The date I was born in MM/DD/YYYY format is:

```python
from datetime import datetime
from dateutil.relativedelta import relativedelta

today = datetime(2023, 2, 27)
birthdate = today - relativedelta(years=25)
print(birthdate.strftime('%m/%d/%Y'))
```

The date I was born in MM/DD/YYYY format is: 02/27/1998
"""
    generatedCode=extractPythonCodeFromResponse(response)
    print(generatedCode)
    exec(generatedCode)