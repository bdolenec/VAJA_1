import person_pb2

# Create a Person object and set fields
person = person_pb2.Person()
person.name = "Alice"
person.age = 30
person.street = "Dunajska"
person.city = "Ljubljana"
person.married = True

# Serialize to file
with open(r"C:\Users\dolenec\Desktop\šola\5 semester\TELEKOMUNIKACIJSKI PROTOKOLI\VAJA_1\VAJE\VAJE-3\DATA\person.pb", "wb") as f:
    f.write(person.SerializeToString())