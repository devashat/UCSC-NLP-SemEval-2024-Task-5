import pandas as pd

# Assuming df is your DataFrame
import pandas as pd

# Assuming df is your DataFrame
df = pd.read_csv('test.csv')


# Strip leading and trailing spaces from 'question' and 'explanation' columns.
df['question'] = df['question'].str.strip()
df['explanation'] = df['explanation'].str.strip()

transformed_df = pd.DataFrame()


# Iterate through each unique question in the DataFrame.
for question, group in df.groupby(['question']):
    
    d = group['answer'].reset_index().to_dict()['answer']
    d[max(d.keys())+1] = "None of The above"
    
    # Create a new row with the question, the explanation with the most words, and the modified answers.
    new_row = {
        'Question': question[0],
        'Explanation': max(group['explanation'], key=lambda x: len(str(x).split())),
        'Answers': d,
    }

    # Append the new row to the transformed DataFrame
    transformed_df = transformed_df._append(new_row, ignore_index=True)

# Save the new Excel file
transformed_df.to_excel('new_test.xlsx', index=False)