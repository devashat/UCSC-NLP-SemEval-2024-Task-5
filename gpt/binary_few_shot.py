from openai import OpenAI
import json
client = OpenAI()
import csv



with open('legal_test.jsonl', 'r') as file:
    # Read lines one by one and process each JSON object
    val_data = [json.loads(line) for line in file]

# Create a new list to store modified rows
modified_data_list = []



# Define the input without the assistant content

ground_truth = []
prediction = []

question1 = "6. Pleading for relief. Giscard sues Munson and Bigby, who both own separate property abutting Giscard‚Äôs. In Count One he claims that construction work Munson has done on his lot has altered the drainage on his property, causing water to flow onto Giscard‚Äôs property and making his lawn soggy. In Count Two, he makes the same allegations against Bigby. He seeks damages from Munson and Bigby for the past interference. He also seeks an injunction against each of them, requiring them to regrade their property to prevent further drainage problems on his."

explanation1 = "Under early English pleading, from which our own pleading rules evolved, a plaintiff was required to stake his case on one version of the facts and law. He could not, for example, plead in one count that the defendant breached a contract with him, and in another that he could recover for fraud. He could not proceed in assumpsit, for breach of contract, and plead replevin, based on a different writ, in the same action. He had to choose a particular theory. Similarly, he had to take one position from the outset about the facts, and stick with it through trial. He couldn’t plead in one count that the defendant acted as an employee of Ace Motor Company at the time of the events in suit, and plead in another claim that he acted as an independent contractor. Common law pleading demanded ruthless efficiency and tough tactical choices. Even if the plaintiff wasn’t sure when he pleaded what the evidence would ultimately show on the issue—and couldn’t be at the time of pleading— he had to commit himself at the outset. A party was required to elect a particular set of facts and a legal theory at the pleading stage. Unfortunately, this forced a litigant to set forth his allegations with a degree of certainty that often was not warranted in terms of the state of his knowledge at that point in the case. If the facts he asserted in the pleadings were not confirmed by later proof, the action or defense would fail even if his proof demonstrated a right to relief on some other theory. Wright & Miller, Federal Practice and Procedure §1282. The Federal Rules rebel against such rigid constraints at the pleading stage. Under Rule 8(d)(3), the plaintiff may assert as many alternative versions of the claim as he has evidence to support, may include both legal and equitable claims in the same complaint, and may assert different versions of a claim ‘‘regardless of consistency.’’ The Rules recognize that, when the plaintiff drafts the complaint, he may not be able to predict what facts will ultimately be found by the jury at trial. For example, it may be impossible at the outset of a case, without discovery, for the plaintiff to know whether a defendant acted as an independent contractor or as an employee. That’s a complex factual issue, which is governed by a fairly ambiguous, multifactor test. See Restatement (Second) of Agency, §220. Ultimately, if the issue must be determined in the case, the jury will decide it based on the evidence presented at trial. The plaintiff’s lawyer isn’t required to stake his case on the prediction of what that determination will be. Under Rule 11, he may allege different versions of his claim, alleging one basis for recovery if the defendant is found to be an employee, and another if he is found to be an independent contractor, so long as he has evidentiary support for each position. Similarly, he may demand different relief, depending on which claim he proves at trial. As the case unfolds, the plaintiff may abandon one version or another of his claim if it becomes clear that one is unsupported. If, however, he has support for multiple versions of his claim after discovery, he may attempt to prove each version of his claim for which he has supporting evidence at trial. Here’s a thought-provoking question that considers the meaning of Rule 8(d)(3)."

Choice = "Giscard has pleaded inconsistently by alleging in one count that Munson caused damage to the lawn, and in the other count that Bigby has."

prompt_1 = f"""

Example 1:
Question:
{question1}

Context:
{explanation1}

Choice: 
{Choice}

"""
ans1 = "0"


question2 = "5. The voice of experience. Ewald sues Stein for injuries suffered when Stein’s sixteen- foot truck rolled down a hill into his car. Ewald alleges, and submits evidence at trial to show that Stein was negligent, because he left the truck parked on a hill, with the emergency brake on, but did not turn the wheels toward the curb. His theory is that the brake was insufficient to hold the truck, so that Stein should have pointed the wheels toward the curb, to prevent it from rolling down the hill into his car. Stein argues that he acted reasonably in relying on the emergency brake. After the parties have presented their proof, the jury is instructed on the basic standard of negligence, and proceeds to its deliberations. During the deliberations, Stepner, one of the jurors, states that he had rented trucks about the size of Stein’s a number of times, and that he always had trouble with the emergency brake on those trucks because the size of the truck requires a very strong brake to hold it. In fact, one time he had a brake let go, causing the truck to roll. The jury renders a verdict for Ewald. A week later, before judgment has entered, Stein’s lawyer runs into one of the other jurors on the street, and the juror mentions Stepner’s comments. Stein’s lawyer files a motion for a new trial, based on the jury’s consideration of extraneous information. The judge should,"

explanation2 = "Another ground for granting a new trial is that the jury, or some of them, were subject to some improper influence that may have affected their deliberations. This could happen in a number of ways. A juror might talk to counsel for one of the parties, or to a party, or to a witness, either in the courthouse or on the street. A trial is a ritualized process, tightly protected from outside influences. The court and counsel orchestrate the production of the evidence the jury is to consider, and test that evidence through presentation and cross-examination. If a juror gets additional evidence from some other source, this untested information may affect the jury, without being subject to scrutiny through the rules of evidence or cross-examination. In re Beverley Hills Supper Club Litigation, 695 F.2d 207 (6th Cir. 1982), provides a dramatic example. In the Beverly Hills Supper Club case, a fire in a night club caused many deaths. A major issue was whether aluminum wiring in the walls had started the fire. The plaintiffs argued that such wiring overheats at the connection to receptacles, creating a risk of fire. After a twenty-two day trial, the jury returned a verdict for the defendant, rejecting this theory. Later a juror wrote to a local newspaper, explaining that the fire couldn’t have happened that way: He had that wiring in his house, and had gone home, taken the plates off the outlets, and found none of the conditions that the plaintiffs argued had caused the fire. Based on this juror experiment, the court ordered a new trial, since the juror had not only conducted the experiment, but had communicated his findings to other jurors. While this seems counterintuitive, barring real world information from the jury, it makes good sense. This evidence may have been very persuasive to the jury, yet improperly so. Who is to say that the juror’s receptacles were the same as those in the night club, or that they were installed in the same way, under the same conditions or using the same wire? Had this evidence been introduced at trial, the plaintiffs’ counsel could have challenged it on such grounds, but if it goes directly to the jury instead, it is not subject to correction or explanation. If a judge finds that extraneous information has found its way into the jury room, she will not automatically grant a new trial motion. Very likely, she will hold a hearing to ascertain how the jury considered the information, whether the evidence was too peripheral to affect the outcome, was simply repetitive of information already before them from the trial, or did not prejudice the party who lost the verdict (where, for example, the extraneous information actually supports the losing party’s case). Only if she is convinced that there is a significant possibility that the improper evidence has affected the jury’s decision will she bite the bullet and order a new trial. Of course, jurors bring a great deal of knowledge and experience with them into the jury room. Jurors do rely on their experience, and this is one of the strengths of the jury system. But there is a distinction— sometimes a fine one—between testing the evidence against one’s general experience, and introducing evidence into the jury’s deliberations that was not placed before the jury by the parties. Here’s a question that explores that distinction."


choices2 = "deny the new trial motion."

prompt_2 = f"""
Example 2:
Question:
{question2}

Context:
{explanation2}

Choice:
{choices2}
"""

ans2 = "1"

system_instruction = """

You are an AI legal expert with expertise in U.S. Civil Procedure and U.S. Civil Law, known for your strong reasoning abilities. 
Your task is to answer a question in the legal domain.

You will be provided with:

1. question: A legal question
2. context: Additional context for better understanding
3. answer candidate: an answer candidate that can be either correct or incorrect

Your response should be a string with length 1. You will be classifying a correct answer as 1, and an incorrect answer as 0.

Your goal is to ensure accurate answers and thorough reasoning.

"""


for row in val_data:
    message = []
    #message.append(row['messages'][0])
    message.append({'role': 'system', 'content': system_instruction})
    message.append({"role": "user", "content": prompt_1})
    message.append({"role": "assistant", "content": ans1})
    message.append({"role": "user", "content": prompt_2})
    message.append({"role": "assistant", "content": ans2})
    
    
    message.append(row['messages'][1])
    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        #model="gpt-3.5-turbo-0125", #"ft:gpt-3.5-turbo-0613:llminterpret::8fKjBh02", #"gpt-3.5-turbo",
        messages=message
    )
    prediction.append(response.choices[0].message.content)

# Writing lists to a CSV file
with open('test_prediction_few_shot.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Prediction'])

    # Zip the lists and write each row to the CSV file
    for row in zip(prediction):
        csv_writer.writerow(row)