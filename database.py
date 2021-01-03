from pymongo import MongoClient

class Database:
    def __init__(self):
        try:
            # self.client = MongoClient('localhost', 27017)
            # self.db = self.client['stdDropoutDB']
            # self.collectionD = self.db['MOOC_Visual']
            self.client = MongoClient("mongodb+srv://gowtham136:user136@cluster0.heyil.mongodb.net/<dbname>?retryWrites=true&w=majority")
            self.db = self.client['stdDropoutDB']
            self.collectionT = self.db['MOOC_Visual']
        except Exception as ex:
            print(ex)

    # To add new row
    def update_record(self, df):
        record = df.to_dict(orient='records')[0]
        self.collectionT.insert_one(record)     # Inserting Record
        countOfrecords = self.collectionT.find().count()    # Finding number of records
        message = f"Record is successfully inserted at place {countOfrecords}"  # Sending Message
        return message