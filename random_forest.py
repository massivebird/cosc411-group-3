#Testing linear regression on new dataset
#Import the required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

class Model:
    def __init__(self):
        #Load data
        data = pd.read_csv('jupyter/testDataset.csv')

        #Find the min and max of each column
        linguisticMin = data['Linguistic'].min()
        linguisticMax = data['Linguistic'].max()

        musicalMin = data['Musical'].min()
        musicalMax = data['Musical'].max()

        bodilyMin = data['Bodily'].min()
        bodilyMax = data['Bodily'].max()

        logMathMin = data['Logical - Mathematical'].min()
        logMathMax = data['Logical - Mathematical'].max()

        spatVisMin = data['Spatial-Visualization'].min()
        spatVisMax = data['Spatial-Visualization'].max()

        interpersonalMin = data['Interpersonal'].min()
        interpersonalMax = data['Intrapersonal'].max()

        intrapersonalMin = data['Intrapersonal'].min()
        intrapersonalMax = data['Intrapersonal'].max()

        naturalistMin = data['Naturalist'].min()
        naturalistMax = data['Naturalist'].max()

        print('Linguistic: ', linguisticMin, linguisticMax)
        print('Musical: ', musicalMin, musicalMax)
        print('Bodily: ', bodilyMin, bodilyMax)
        print('Logical-Mathematical: ', logMathMin, logMathMax)
        print('Spatial-Visual: ', spatVisMin, spatVisMax)
        print('Interpersonal: ', interpersonalMin, interpersonalMax)
        print('Intrapersonal: ', intrapersonalMin, intrapersonalMax)
        print('Naturalist: ', naturalistMin, naturalistMax)

        #Convert the profession to numbers
        encoder = LabelEncoder()
        self.encoder = encoder
        data['Job profession'] = encoder.fit_transform(data['Job profession'])

        #Normalize the data
        columns_to_normalize = data.columns.difference(['Job profession'])

        scaler = MinMaxScaler()

        data[columns_to_normalize] = pd.DataFrame(scaler.fit_transform(data[columns_to_normalize]))

        #Separate into features X and target Y
        X = data.drop('Job profession', axis=1)
        Y = data['Job profession']

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        #Initialize the random forest classifier
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier = classifier
        classifier.fit(X_train, Y_train)

        #Make predictions on the test set
        y_pred = classifier.predict(X_test)
        print(y_pred)

    # Returns the name of a career based on the input values.
    def predict(self, vals):
        pred_id = self.classifier.predict(vals)
        return self.encoder.inverse_transform(pred_id)

def model():
    return Model()
