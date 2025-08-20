# django-nlp-filters

`django-nlp-filters` is a Python package that allows you to perform **natural language-based filtering** on Django QuerySets.  
With this library, you can write queries like:

- "get all users whose first name is Aravind"  
- "filter employees where age greater than 30"  
- "find orders with status pending and amount less than 1000"

and have them translated into Django ORM filters automatically.

---

## 🚀 Features

- Natural language to Django ORM filter conversion.
- Works with any Django model.
- Extensible for custom operators and field mappings.
- Lightweight, no external heavy NLP dependencies.

---

## 📦 Installation

```bash
pip install django-nlp-filters

⚡ Quickstart
1. Import and use in your Django project:
from nlp_filters.parser import NLPFilter

# Example with Django's User model
from django.contrib.auth.models import User

nlp_filter = NLPFilter(User)

query = "get all users whose first name is Aravind"
qs = nlp_filter.filter(query)

print(qs.query)


Output SQL equivalent:

SELECT * FROM auth_user WHERE first_name = 'Aravind';

🔧 Usage Examples
from nlp_filters.parser import NLPFilter
from myapp.models import Employee

nlp = NLPFilter(Employee)

# Example 1: equality filter
qs = nlp.filter("get all employees whose age is 30")

# Example 2: greater than
qs = nlp.filter("get all employees whose salary greater than 50000")

# Example 3: AND condition
qs = nlp.filter("get all employees whose age greater than 25 and department is IT")

# Example 4: OR condition
qs = nlp.filter("get all employees whose city is Kochi or city is Bangalore")

⚙️ How It Works

The query string is parsed into tokens (fields, operators, values).

Mapped to Django field lookups (exact, icontains, gt, lt, etc.).

Generates Q objects that combine filters using & (AND) or | (OR).

Applies them on the provided model’s queryset.

🧩 Supported Operators
Natural Language	Django Lookup
"is", "equals", "equal to"	exact
"contains"	icontains
"greater than"	gt
"less than"	lt
"greater than or equal to"	gte
"less than or equal to"	lte
🛠️ Extending the Library

You can add new operators or custom field mappings:

from nlp_filters.parser import NLPFilter

nlp = NLPFilter(MyModel)

# Add custom operator
nlp.operators["starts with"] = "istartswith"

# Add synonym for a field
nlp.field_map["mobile"] = "phone_number"

📂 Project Structure

django-nlp-filters/
├── nlp_filters/
│   ├── __init__.py
│   ├── parser.py   # main library code
├── setup.py
├── README.md
├── LICENSE
├── MANIFEST.in

🤝 Contributing

Contributions are welcome! 🎉

Fork the repo

Create your feature branch (git checkout -b feature/my-feature)

Commit changes (git commit -m 'Add feature')

Push to branch (git push origin feature/my-feature)

Open a Pull Request

📜 License

This project is licensed under the MIT License - see the LICENSE file for details.


---

