import pandas as pd

# Assuming df is your DataFrame
import pandas as pd

# Assuming df is your DataFrame
df = pd.read_csv('test.csv')

df['question'] = df['question'].str.strip()
df['explanation'] = df['explanation'].str.strip()
transformed_df = pd.DataFrame()

for question, group in df.groupby(['question']):
    
    # if any(group['label'] == 1):
    #     # Extract the correct answer for the current question
    #     correct_answer = group.loc[group['label'] == 1, 'answer'].iloc[0]
    # else:
    #     # If no answer has label 1, set correct_answer to an empty string or any default value
    #     correct_answer = ''

    # Create a new row for the question in the transformed DataFrame
    #print("GOUTP ANS: ", group['answer'].values)
    d = group['answer'].reset_index().to_dict()['answer']
    d[max(d.keys())+1] = "None of The above"
    new_row = {
        'Question': question[0],
        'Explanation': max(group['explanation'], key=lambda x: len(str(x).split())),
        'Answers': d,#'[' + ', '.join(map(str, group['answer'])) + ']',
        #'Correct_Answer': correct_answer,
        #'Analysis': '<sep>'.join(map(str, group['analysis']))
    }

    # Append the new row to the transformed DataFrame
    transformed_df = transformed_df._append(new_row, ignore_index=True)

# Save the new Excel file
transformed_df.to_excel('new_test.xlsx', index=False)