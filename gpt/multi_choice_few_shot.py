from openai import OpenAI
import json
import pandas as pd
from difflib import SequenceMatcher

client = OpenAI()

new_train_data = pd.read_excel('new_test.xlsx')
train_data = pd.read_csv('test.csv')

# Define instructions for the AI model, emphasizing the requirement for accurate answers and thorough reasoning.
system_instruction = """
You are an AI legal expert with expertise in U.S. Civil Procedure and U.S. Civil Law, known for your strong reasoning abilities. Your task is to answer a Multiple Choice Question in the legal domain. Choose an answer only if you are very confident, otherwise, select "None of The Above."

You will be provided with:
1. question: A legal question
2. context: Additional context for better understanding
3. choices: Multiple answer candidates

Your response should be a JSON with two keys: "correct_answer" and "reasoning." Place the correct answer exactly as provided in the "correct_answer" key. Provide a detailed explanation of your reasoning in the "reasoning" key. Do not add or remove any other text.

Your goal is to ensure accurate answers and thorough reasoning.
"""


# Example questions, explanations, and choices are defined here for the few-shot learning setup.
question1 = "6. Pleading for relief. Giscard sues Munson and Bigby, who both own separate property abutting Giscard‚Äôs. In Count One he claims that construction work Munson has done on his lot has altered the drainage on his property, causing water to flow onto Giscard‚Äôs property and making his lawn soggy. In Count Two, he makes the same allegations against Bigby. He seeks damages from Munson and Bigby for the past interference. He also seeks an injunction against each of them, requiring them to regrade their property to prevent further drainage problems on his."

explanation1 = "Under early English pleading, from which our own pleading rules evolved, a plaintiff was required to stake his case on one version of the facts and law. He could not, for example, plead in one count that the defendant breached a contract with him, and in another that he could recover for fraud. He could not proceed in assumpsit, for breach of contract, and plead replevin, based on a different writ, in the same action. He had to choose a particular theory. Similarly, he had to take one position from the outset about the facts, and stick with it through trial. He couldn’t plead in one count that the defendant acted as an employee of Ace Motor Company at the time of the events in suit, and plead in another claim that he acted as an independent contractor. Common law pleading demanded ruthless efficiency and tough tactical choices. Even if the plaintiff wasn’t sure when he pleaded what the evidence would ultimately show on the issue—and couldn’t be at the time of pleading— he had to commit himself at the outset. A party was required to elect a particular set of facts and a legal theory at the pleading stage. Unfortunately, this forced a litigant to set forth his allegations with a degree of certainty that often was not warranted in terms of the state of his knowledge at that point in the case. If the facts he asserted in the pleadings were not confirmed by later proof, the action or defense would fail even if his proof demonstrated a right to relief on some other theory. Wright & Miller, Federal Practice and Procedure §1282. The Federal Rules rebel against such rigid constraints at the pleading stage. Under Rule 8(d)(3), the plaintiff may assert as many alternative versions of the claim as he has evidence to support, may include both legal and equitable claims in the same complaint, and may assert different versions of a claim ‘‘regardless of consistency.’’ The Rules recognize that, when the plaintiff drafts the complaint, he may not be able to predict what facts will ultimately be found by the jury at trial. For example, it may be impossible at the outset of a case, without discovery, for the plaintiff to know whether a defendant acted as an independent contractor or as an employee. That’s a complex factual issue, which is governed by a fairly ambiguous, multifactor test. See Restatement (Second) of Agency, §220. Ultimately, if the issue must be determined in the case, the jury will decide it based on the evidence presented at trial. The plaintiff’s lawyer isn’t required to stake his case on the prediction of what that determination will be. Under Rule 11, he may allege different versions of his claim, alleging one basis for recovery if the defendant is found to be an employee, and another if he is found to be an independent contractor, so long as he has evidentiary support for each position. Similarly, he may demand different relief, depending on which claim he proves at trial. As the case unfolds, the plaintiff may abandon one version or another of his claim if it becomes clear that one is unsupported. If, however, he has support for multiple versions of his claim after discovery, he may attempt to prove each version of his claim for which he has supporting evidence at trial. Here’s a thought-provoking question that considers the meaning of Rule 8(d)(3)."

choices1 = "{0: 'Giscard has pleaded inconsistently by alleging in one count that Munson caused damage to the lawn, and in the other count that Bigby has.', 1: 'Giscard has pleaded inconsistently by seeking damages for the past interference, but an injunction against future interference.', 2: 'Giscard has pleaded inconsistently both with regard to the defendant who caused the harm and with regard to the type of relief, but this is permissible under Rule 8(d)(3).'}"

prompt_1 = f"""

Example 1:
Question:
{question1}

Context:
{explanation1}

Choices:
{choices1}
"""


ans1 = """
{   
    "correct_answer": "None of The above",
    "reasoning": "This question's a bit droll, but makes a point. There's nothing inconsistent about this pleading. Giscard has alleged that Munson damaged his property by altering the drainage pattern, and that Bigby did. It's entirely possible that he will prove that they both did. Liability here isn't an 'either/or' proposition, in which, if he recovers against one defendant, he will definitely lose against the other. Both may have contributed to Giscard's soggy lawn problem, and both may be liable to him if they did. Giscard's demands for relief aren't inconsistent either. There's nothing inconsistent about demanding damages for past interference and protection from future interference. In fact, these demands are utterly 'consistent,' aren't they? This is why None of the given choices are correct. "
}
"""

question2 = "5. The voice of experience. Ewald sues Stein for injuries suffered when Stein’s sixteen- foot truck rolled down a hill into his car. Ewald alleges, and submits evidence at trial to show that Stein was negligent, because he left the truck parked on a hill, with the emergency brake on, but did not turn the wheels toward the curb. His theory is that the brake was insufficient to hold the truck, so that Stein should have pointed the wheels toward the curb, to prevent it from rolling down the hill into his car. Stein argues that he acted reasonably in relying on the emergency brake. After the parties have presented their proof, the jury is instructed on the basic standard of negligence, and proceeds to its deliberations. During the deliberations, Stepner, one of the jurors, states that he had rented trucks about the size of Stein’s a number of times, and that he always had trouble with the emergency brake on those trucks because the size of the truck requires a very strong brake to hold it. In fact, one time he had a brake let go, causing the truck to roll. The jury renders a verdict for Ewald. A week later, before judgment has entered, Stein’s lawyer runs into one of the other jurors on the street, and the juror mentions Stepner’s comments. Stein’s lawyer files a motion for a new trial, based on the jury’s consideration of extraneous information. The judge should,"

explanation2 = "Another ground for granting a new trial is that the jury, or some of them, were subject to some improper influence that may have affected their deliberations. This could happen in a number of ways. A juror might talk to counsel for one of the parties, or to a party, or to a witness, either in the courthouse or on the street. A trial is a ritualized process, tightly protected from outside influences. The court and counsel orchestrate the production of the evidence the jury is to consider, and test that evidence through presentation and cross-examination. If a juror gets additional evidence from some other source, this untested information may affect the jury, without being subject to scrutiny through the rules of evidence or cross-examination. In re Beverley Hills Supper Club Litigation, 695 F.2d 207 (6th Cir. 1982), provides a dramatic example. In the Beverly Hills Supper Club case, a fire in a night club caused many deaths. A major issue was whether aluminum wiring in the walls had started the fire. The plaintiffs argued that such wiring overheats at the connection to receptacles, creating a risk of fire. After a twenty-two day trial, the jury returned a verdict for the defendant, rejecting this theory. Later a juror wrote to a local newspaper, explaining that the fire couldn’t have happened that way: He had that wiring in his house, and had gone home, taken the plates off the outlets, and found none of the conditions that the plaintiffs argued had caused the fire. Based on this juror experiment, the court ordered a new trial, since the juror had not only conducted the experiment, but had communicated his findings to other jurors. While this seems counterintuitive, barring real world information from the jury, it makes good sense. This evidence may have been very persuasive to the jury, yet improperly so. Who is to say that the juror’s receptacles were the same as those in the night club, or that they were installed in the same way, under the same conditions or using the same wire? Had this evidence been introduced at trial, the plaintiffs’ counsel could have challenged it on such grounds, but if it goes directly to the jury instead, it is not subject to correction or explanation. If a judge finds that extraneous information has found its way into the jury room, she will not automatically grant a new trial motion. Very likely, she will hold a hearing to ascertain how the jury considered the information, whether the evidence was too peripheral to affect the outcome, was simply repetitive of information already before them from the trial, or did not prejudice the party who lost the verdict (where, for example, the extraneous information actually supports the losing party’s case). Only if she is convinced that there is a significant possibility that the improper evidence has affected the jury’s decision will she bite the bullet and order a new trial. Of course, jurors bring a great deal of knowledge and experience with them into the jury room. Jurors do rely on their experience, and this is one of the strengths of the jury system. But there is a distinction— sometimes a fine one—between testing the evidence against one’s general experience, and introducing evidence into the jury’s deliberations that was not placed before the jury by the parties. Here’s a question that explores that distinction."


choices2 = "{0: 'grant the motion for a new trial.', 1: 'conduct a hearing to determine whether the jury’s decision was influenced by Stepner’s information.', 2: 'grant a new trial if she believes the information may have influenced the other jurors in reaching a verdict.', 3: 'deny the new trial motion.'}"

prompt_2 = f"""
Example 2:
Question:
{question2}

Context:
{explanation2}

Choices:
{choices2}
"""

ans2 = """
{   
    "correct_answer": "deny the new trial motion.",
    "reasoning": "This is the best answer. Stepner's consideration of his own experience, and sharing that experience with the other jurors, is a permissible part of the jury's deliberations, and would not support the grant of a new trial."
}
"""


# Function to format the question, context, and choices, then query the GPT model.
def analyse_reviews(question, explanation, choices):

    prompt = f"""
    Question:
    {question}

    Context:
    {explanation}

    Choices:
    {choices}
    """
    completion = client.chat.completions.create(
        #model="gpt-4-0125-preview",
        model="gpt-3.5-turbo-1106",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt_1},
            {"role": "assistant", "content": ans1},
            {"role": "user", "content": prompt_2},
            {"role": "assistant", "content": ans2},
            {"role": "user", "content": prompt},
        ]
    )
    try:
        generated_text = completion.choices[0].message.content
        print(f'PRED: {json.loads(generated_text)["correct_answer"]}')
        return json.loads(generated_text)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


new_df = pd.DataFrame({
    'index': train_data['idx'],
    'question': train_data['question'], 
    'answer': train_data['answer']
})
new_df['predicted_label'] = ""
new_df['generated_Ans'] = ""


# Function to calculate the similarity between two strings.
def calculate_similarity(str1, str2):
    seq_matcher = SequenceMatcher(None, str1.strip(), str2.strip())
    similarity_score = seq_matcher.ratio()
    return similarity_score


# Iterate over each row in the new test data to predict answers and calculate similarity.
for index, row in new_train_data.iterrows():
    question = row['Question'].strip() 
    explanation = row['Explanation'] 
    choices = row['Answers']  

    # Call the function for each row
    result = analyse_reviews(question, explanation, choices)

    
    print("----------")
    if result:
        for index1, row1 in new_df.iterrows():
            question1 = row1['question']
            
            if question == row1['question']: 
                #print("--")
                new_df.at[index1, 'generated_Ans'] = result['correct_answer']
                if row1['answer'].strip().lower()  == result['correct_answer'].strip().lower():
                     
                    print("-----", 1)
                    new_df.at[index1, 'predicted_label'] = 1


# Save the updated DataFrame to a new CSV file
new_df.to_csv('pred_test.csv', index=False)