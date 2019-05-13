import pandas as pd


def make_submission_csv(test, output, prediction_classes):
    output_df = pd.DataFrame(output, columns = prediction_classes)
    output_df_upload = test.join(output_df)
    output_df_upload = output_df_upload[['id'] + prediction_classes]
    print('Writing predictions to csv file')
    output_df_upload.to_csv('submission.csv', index = False)
    print('File written and ready to be uploaded')