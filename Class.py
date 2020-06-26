class Employee:
    raise_amount = 1.04
    # Creating he constructor(initializer)
    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first + "." + last + "@company.com"

    # A method of our class
    # self: positional argument
    def fullname(self):
        return "{} {}".format(self.first, self.last)

    def apply_raise(self):
        self.pay = int(self.pay * Employee.raise_amount)


# instances of class
emp_1 = Employee("Onur", "Surucu", 20000)
emp_2 = Employee("Uygar", "Yarak", 10000)

Employee.fullname(emp_1)

# Prints where is the instance assigned to
print(emp_1)
# Printing out the name space of emp_1
print(emp_1.__dict__)
# Printing the first and last names
print("{} {}".format(emp_1.first, emp_1.last))

# Applying raise
print(emp_1.pay)
emp_1.apply_raise()
print(emp_1.pay)

