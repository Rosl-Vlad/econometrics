import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

from scipy import stats
from pyreadstat import read_sav, set_value_labels
import matplotlib.pyplot as plt


from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
sns.set_theme()

data, meta = read_sav("r29i_os_42.sav")
select_columns = [
    "yj10",     # wage
    "yj1.1.1",  # job satisfaction
    "yj4.1",    # job 
    "y_diplom", # education
    "yj260",    # foreign language
    "ym80",     # alcohol consumption
    "y_age",    # age
    "yj161.3y", # job experience
    "psu",      # region of residence 
    "yj6",      # number of subordinates
    "yj6.1a",   # work time
    "yj401.1a", # OK user
    "yj401.2a", # VK user
    "yj401.3a", # FaceBook user 
    "yj401.5a", # Twitter user 
    "yh5",      # sex
]
# Рассмотрим только работающих людей (data["yj1"] == 1) и отфильтруем данные с пропусками 

df_w = data[
    (data["yj1"] == 1) 
    & (data["yj10"] < 99999996) 
    & (data["yj1.1.1"] < 99999996) 
    & (data["yj4.1"] < 99999996)
    & (data["y_diplom"] < 99999996)
    & (data["yj260"] < 99999996)
    & (data["ym80"] < 99999996)
    & (data["y_age"] < 99999996)
    & (data["yj161.3y"] < 99999996)
    & (data["yj6"] < 99999996)
    & (data["yj6.1a"] < 99999996)
    & (data["psu"] < 99999996)
    & (data["yj401.1a"] < 99999996)
    & (data["yj401.2a"] < 99999996)
    & (data["yj401.3a"] < 99999996)
    & (data["yj401.5a"] < 99999996)
    & (data["yh5"] < 99999996)
    ][select_columns]


for col in ["yj4.1", "y_diplom", "yj260", "ym80", "psu", "yh5", "yj401.1a", "yj401.2a", "yj401.3a", "yj401.5a"]:
    df_w[col] = pd.Categorical(df_w[col])
    
d = {"yj10":    "wage",
    "yj1.1.1":  "job_satisfaction",
    "y_diplom": "education",
    "yj4.1":    "job",
    "yj260":    "foreigin_lan",
    "ym80":     "alc",
    "y_age":    "age",
    "yj161.3y": "job_experience",
    "yj6":      "supervisor",
    "yj6.1a":   "work_time",
    "yj401.1a": "ok",
    "yj401.2a": "vk",
    "yj401.3a": "facebook",
    "yj401.5a": "twitter",
    "yh5":      "male"
    }
df_tmp = df_w.rename(columns = d)

sns.set(rc={'figure.figsize':(10,8)})

sns.histplot(df_tmp["wage"])
plt.xlabel("Wage")
plt.title("Wage distribution")
plt.savefig('Wage distribution.png')
plt.show()

sns.set(rc={'figure.figsize':(10,8)})

df_tmp_1 = df_tmp[df_tmp["wage"] < 70000]
di = {1: "ПОЛНОСТЬЮ УДОВЛЕТВОРЕНЫ",
      2: "СКОРЕЕ УДОВЛЕТВОРЕНЫ",
      3: "ИДА,И НЕТ",
      4: "СКОРЕЕ НЕ УДОВЛЕТВОРЕНЫ",
      5: "СОВСЕМ НЕ УДОВЛЕТВОРЕНЫ"}
df_tmp_1 = df_tmp_1.replace({"job_satisfaction": di})


sns.histplot(df_tmp_1["job_satisfaction"])
plt.xlabel("Job satisfaction")
plt.xticks(rotation=45)
plt.title("Job satisfaction distribution")
plt.savefig('Job satisfaction distribution.png')
plt.show()

sns.set(rc={'figure.figsize':(10,8)})

df_tmp_1 = df_tmp[df_tmp["wage"] > -70000]
di = {1: "окончил 0 - 6 классов",
      2: "незаконченное среднее \n образование (7 - 8 кл)",
      3: "незаконченное среднее \n образование (7 - 8 кл) + что-то еще",
      4: "законченное среднее образование",
      5: "законченное среднее специальное образование",
      6: "законченное высшее образование и выше"
     }
df_tmp_1 = df_tmp_1.replace({"education": di})


sns.histplot(df_tmp_1["education"])
plt.xlabel("Education")
plt.xticks(rotation=90)
plt.title("Education distribution")
plt.savefig('Education distribution.png')
plt.show()

df_tmp_1 = df_tmp[df_tmp["wage"] < 70000]
sns.histplot(df_tmp_1["job"].cat.as_ordered())
plt.xlabel("Job")
plt.xticks(rotation=0)
plt.title("Job distribution")
plt.savefig('Job distribution.png')
plt.show()

df_tmp_1 = df_tmp[df_tmp["wage"] < 70000]

di = {1: "Да",
      2: "Нет",
     }
df_tmp_1 = df_tmp_1.replace({"foregin_lan": di})

sns.histplot(df_tmp_1["foregin_lan"], discrete = True)
plt.xlabel("Foreign language")
plt.xticks(rotation=0)
plt.title("Foreign language distribution")
plt.savefig('Foreign language distribution.png')
plt.show()

df_tmp_1 = df_tmp[df_tmp["wage"] < 70000]

di = {1: "Да",
      2: "Нет",
     }
df_tmp_1 = df_tmp_1.replace({"alc": di})

sns.histplot(df_tmp_1["alc"], discrete = True)

plt.xlabel("Alcohol consumption")
plt.xticks(rotation=0)
plt.title("Alcohol consumption distribution")
plt.savefig('Alcohol consumption distribution.png')
plt.show()

df_tmp_1 = df_tmp[df_tmp["wage"] < 70000]

sns.histplot(df_tmp_1["age"], discrete = True)

plt.xlabel("Age")
plt.xticks(rotation=0)
plt.title("Age distribution")
plt.savefig('Age distribution.png')
plt.show()

df_tmp_1 = df_tmp[df_tmp["wage"] < 70000]

sns.histplot(df_tmp_1["job_experience"], discrete = True)

plt.xlabel("Job experience")
plt.xticks(rotation=0)
plt.title("Job experience distribution")
plt.savefig('Job experience distribution.png')
plt.show()

df_tmp_1 = df_tmp[df_tmp["wage"] < 70000]



di = {1: "Да",
      2: "Нет",
     }
df_tmp_1 = df_tmp_1.replace({"supervisor": di})

sns.histplot(df_tmp_1["supervisor"], discrete = True)

plt.xlabel("Supervisor")
plt.xticks(rotation=0)
plt.title("Supervisor distribution")
plt.savefig('Supervisor distribution.png')
plt.show()

df_tmp_1 = df_tmp[df_tmp["wage"] < 70000]

sns.histplot(df_tmp_1["work_time"], discrete = True)

plt.xlabel("Work time")
plt.xticks(rotation=0)
plt.title("Work time distribution")
plt.savefig('Work time distribution.png')
plt.show()

df_tmp_1 = df_tmp[df_tmp["wage"] < 70000]

di = {1: "Да",
      2: "Нет",
     }
df_tmp_1 = df_tmp_1.replace({"ok": di})

sns.histplot(df_tmp_1["ok"], discrete = True)
plt.xlabel("Usage of OK")
plt.xticks(rotation=0)
plt.title("Usage of OK distribution")
plt.savefig('Usage of OK distribution.png')
plt.show()

df_tmp_1 = df_tmp[df_tmp["wage"] < 70000]

di = {1: "Да",
      2: "Нет",
     }
df_tmp_1 = df_tmp_1.replace({"vk": di})

sns.histplot(df_tmp_1["vk"], discrete = True)
plt.xlabel("Usage of VK")
plt.xticks(rotation=0)
plt.title("Usage of VK distribution")
plt.savefig('Usage of VK distribution.png')
plt.show()

df_tmp_1 = df_tmp[df_tmp["wage"] < 70000]

di = {1: "Да",
      2: "Нет",
     }
df_tmp_1 = df_tmp_1.replace({"facebook": di})

sns.histplot(df_tmp_1["facebook"], discrete = True)
plt.xlabel("Usage of facebook")
plt.xticks(rotation=0)
plt.title("Usage of facebook distribution")
plt.savefig('Usage of facebook distribution.png')
plt.show()

df_tmp_1 = df_tmp[df_tmp["wage"] < 70000]

di = {1: "Да",
      2: "Нет",
     }
df_tmp_1 = df_tmp_1.replace({"twitter": di})

sns.histplot(df_tmp_1["twitter"], discrete = True)
plt.xlabel("Usage of twitter")
plt.xticks(rotation=0)
plt.title("Usage of twitter distribution")
plt.savefig('Usage of twitter distribution.png')
plt.show()

df_tmp_1 = df_tmp[df_tmp["wage"] < 70000]

di = {1: "Мужской",
      2: "Женский",
     }
df_tmp_1 = df_tmp_1.replace({"male": di})

sns.histplot(df_tmp_1["male"], discrete = True)
plt.xlabel("Sex")
plt.xticks(rotation=0)
plt.title("Sex distribution")
plt.savefig('Sex distribution.png')
plt.show()

df_tmp_1 = df_tmp[df_tmp["wage"] < 70000]
di = {1: "ПОЛНОСТЬЮ УДОВЛЕТВОРЕНЫ",
      2: "СКОРЕЕ УДОВЛЕТВОРЕНЫ",
      3: "ИДА,И НЕТ",
      4: "СКОРЕЕ НЕ УДОВЛЕТВОРЕНЫ",
      5: "СОВСЕМ НЕ УДОВЛЕТВОРЕНЫ"}
df_tmp_1 = df_tmp_1.replace({"job_satisfaction": di})

sns.histplot(data=df_tmp_1, x="wage", hue="job_satisfaction", multiple="fill")
plt.xlabel("Wage")
plt.ylabel("%")
plt.title("Wage distribution with job satisfaction index")
plt.savefig('1. Wage distribution with job satisfaction index.png')
plt.show()

df_tmp_1 = df_tmp[df_tmp["wage"] < 150000]

di = {1: "0-6 классов",
      2: "незаконченное среднее образование",
      3: "незаконченное среднее образование + что-то еще",
      4: "законченное среднее образование",
      5: "законченное среднее специальное образование",
      6: "законченное высшее образование и выше"}
df_tmp_1 = df_tmp_1.replace({"education": di})

sns.histplot(data=df_tmp_1, x="wage", hue="education", multiple="fill", shrink=.8, bins=15)
plt.xlabel("Wage")
plt.ylabel("%")
plt.title("Wage distribution with education")
plt.savefig('2. Wage distribution with education.png')
plt.show()

df_tmp_1 = df_tmp[df_tmp["wage"] < 150000]

di = {1: "Да",
      2: "Нет"}
df_tmp_1 = df_tmp_1.replace({"alc": di})

sns.histplot(data=df_tmp_1, x="wage", hue="alc", multiple="fill", shrink=.8, bins=15)
plt.xlabel("Wage")
plt.ylabel("%")
plt.title("Wage distribution with alcohol consumption")
plt.savefig('2. Wage distribution with alcohol consumption.png')
plt.show()

df_tmp_1 = df_tmp[df_tmp["wage"] < 150000]

di = {1: "Да",
      2: "Нет"}
df_tmp_1 = df_tmp_1.replace({"supervisor": di})

sns.histplot(data=df_tmp_1, x="wage", hue="supervisor", multiple="fill", shrink=.8, bins=15)
plt.xlabel("Wage")
plt.ylabel("%")
plt.title("Wage distribution with supervisor")
plt.savefig('2. Wage distribution with supervisor.png')
plt.show()
df_tmp_1 = df_tmp[df_tmp["wage"] < 150000]

def fff(s):
    if s == 0:
        return 0
    if s > 0 and s <= 5:
        return 1
    if s >= 6 and s <= 10:
        return 2
    if s > 10 and s <= 15:
        return 3
    return 4
    
    

di = {0: "0",
      1: "[1;5]",
      2: "[6;10]",
      3: "(10;15]",
      4: "(15;inf]",
     }
df_tmp_1["job_experience"] = df_tmp["job_experience"].apply(lambda s: fff(s))
df_tmp_1 = df_tmp_1.replace({"job_experience": di})

sns.histplot(data=df_tmp_1, x="wage", hue="job_experience", multiple="fill", shrink=.8, bins=15)
plt.xlabel("Wage")
plt.ylabel("%")
plt.title("Wage distribution with Job experience")
plt.savefig('2. Wage distribution with Job experience.png')
plt.show()

df_tmp_1 = df_tmp[df_tmp["wage"] < 150000]

def fff(s):
    if s > 0 and s <= 20:
        return 1
    if s >= 20 and s <= 25:
        return 2
    if s > 25 and s <= 35:
        return 3
    if s > 35 and s <= 45:
        return 4
    return 5
    
    

di = {1: "[0;20]",
      2: "(20;25]",
      3: "(25;35]",
      4: "(35;45]",
      5: "(45, inf)"
     }
df_tmp_1["age"] = df_tmp["age"].apply(lambda s: fff(s))
df_tmp_1 = df_tmp_1.replace({"age": di})

sns.histplot(data=df_tmp_1, x="wage", hue="age", multiple="fill", shrink=.8, bins=15)
plt.xlabel("Wage")
plt.ylabel("%")
plt.title("Wage distribution with age")
plt.savefig('2. Wage distribution with age.png')
plt.show()