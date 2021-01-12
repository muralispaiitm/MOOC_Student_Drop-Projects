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

    # To update the record
    def update_record(self, df):
        input_record = df.to_dict(orient='records')[0]
        # ----------- Verifying the above record is existing in the database --------------
        for rec in self.collectionT.find():
            if list(input_record.values()) == list(rec.values())[1:]:
                message = f"Record is already presenting in the database at index {list(rec.values())[0]}"
                return message

        # -------- Inserting the above record into database if it is not presenting in the database -------
        countOfrecords = self.collectionT.find().count()  # Finding number of records
        record = {"_id": countOfrecords+1}
        record.update(input_record)
        self.collectionT.insert_one(record)     # Inserting Record
        message = f"Record is successfully inserted at place {countOfrecords+1}"  # Sending Message
        return message