from sqlite3 import *
import uuid
import time

db = connect('CORE.db')
cursor = db.cursor()

valid = 1

uuidGen = ""

Uarray = []

end = True

def exec_sql():

    while end:
        
            global end
            global err
            
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
                err = True
return

def exec_break():
    if UUID == 'q':
            db.close()
            end = False
            print("Connection closed.\n")
return


if err:
    exec_sql()
    else:
        print(valid)

UUID = input("Create UUID? y/yes (or 'q' to quit): ")

if UUID == "y" or UUID == "yes":
    end = True
    exec_sql()
    
elif UUID == "q":
    
    exec_break()



