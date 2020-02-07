

class student:
    raise_amt = 1.04 # instant variable
    def __init__(self, pay, name=None,j = 10, age=None):
        self.name = name
        # self.age = age
        self.age = age if age is not None else j
        self.pay = pay

    variable = 1
    # Regular method # automatically take the instance (i.e. S1,S2) as the first input
    def method(self):
        return self.pay * self.raise_amt

    # class method # To work with CLASS information # NOT automatically take the instance (i.e. S1,S2) as the first input
    @classmethod    # <----- Decorator
    def classmethod(cls, amount): # clc is class variable
        cls.raise_amt = amount

    @staticmethod
    def staticmethod():
        return 'static method called'



S1 = student(2000,"ShihLuen", 12,30)
S2 = student(3000,"Seyed", 12,35)

S1.classmethod(1.09)

print(S1.method())

S1.raise_amt = 1.05
print(student.raise_amt)
print(S1.raise_amt)
print(S2.raise_amt)

student.raise_amt = 1.06

print(S1.raise_amt)
print(S2.raise_amt)

# print(S2.classmethod())

# print(S1.method())

#print(foo.method())
# print(foo.classmethod())

# print(student.classmethod())

# print(student.staticmethod())



# studentlist = []
# studentlist.append(student("ShihLuen", 12,30))
# studentlist.append(student("Seyed", 12,35))


# def printstudents1():
#     for i in studentlist:
#         i.age = i.age +1
#
# printstudents1()
#
# def printstudents():
#     for i in range (len(studentlist)):
#         print('Name: '+studentlist[i].name+' Age:'+str(studentlist[i].age)+'\n')
#
# printstudents()














# def printstudents2():
#     for i in range (0,2):
#         print('Name: '+studentlist[i].name+' Age:'+str(studentlist[i].age)+'\n')
#
# printstudents2()






# def printstudents():
#     for i in range (2):
#         print('Name = '+student1.name+' Age:'+ student1.age)
#



# class X:
#
#     var1 = 1
#     var2 = 2
#
# def foo():
#     print(student2.name)
#     print(student2.age)
# foo()
#
#
# def foo1():
#     print(X.var1)
# foo1()

