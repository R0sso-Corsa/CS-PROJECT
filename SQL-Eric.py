from sqlite3 import *
import uuid
import time

db = connect('CORE.db')
cursor = db.cursor()



uuidGen = ""

Uarray = []

end = True

def exec_sql():

    

    while end:
    
    
        if UUID == "y" or UUID == "yes":
            print("Randomizing UUID...")
        
            for i in range(0,5):
                uuidGen = str(uuid.uuid4())
                Uarray.append(uuidGen)
                print("\n",Uarray[i],"\n")
            
            pick = int(input("Choose generated UUID; 1-5: "))
        
            print("Selected UUID:", Uarray[pick-1])
        
            user = input("Enter username: ")
        
            password = input("Enter password: ")
        
            sql = "INSERT INTO users (UUID, User, Pass) VALUES (?, ?, ?)"
            val = (uuidGen, user, password)
            try:
                cursor.execute(sql, val)
                db.commit()
                print("User added.\n")
            except db.IntegrityError:
                print("Error: Username already exists. Please choose a different username.\n")
            
        if UUID == 'q':
            db.close()
        
            print("Connection closed.\n")
            break

    return




