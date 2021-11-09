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

# print(df_tmp.describe()) # description

df_tmp_ = df_tmp.copy()

conversion_education = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1}
df_tmp_['education'] = df_tmp['education'].replace(conversion_education)

conversion_foregin_lan = {1: 0, 2: 1}
df_tmp['foreigin_lan'] = df_tmp['foreigin_lan'].replace(conversion_foregin_lan)

conversion_alc = {1: 0, 2: 1}
df_tmp['alc'] = df_tmp['alc'].replace(conversion_alc)

#conversion_job = {k: 0 for i, k in enumerate(df_tmp['job'])}
#conversion_job[27] = 1
#conversion_job[31] = 1
#conversion_job[15] = 1
#conversion_job[4] = 1
#df_tmp_['job'] = df_tmp['job'].replace(conversion_job)

conversion_psu = {k: 0 for i, k in enumerate(df_tmp['psu'])}
conversion_psu[2] = 1
df_tmp['psu'] = df_tmp['psu'].replace(conversion_psu) # set binary Moscow or not

conversion_supervisor = {1: 1, 2: 0}
df_tmp['supervisor'] = df_tmp['supervisor'].replace(conversion_supervisor)

conversion_ok= {1: 1, 2: 0}
df_tmp['ok'] = df_tmp['ok'].replace(conversion_ok)

conversion_vk= {1: 1, 2: 0}
df_tmp['vk'] = df_tmp['vk'].replace(conversion_vk)

conversion_facebook= {1: 1, 2: 0}
df_tmp['facebook'] = df_tmp['facebook'].replace(conversion_facebook)

conversion_twitter= {1: 1, 2: 0}
df_tmp['twitter'] = df_tmp['twitter'].replace(conversion_twitter)

conversion_male= {1: 1, 2: 0}
df_tmp['male'] = df_tmp['male'].replace(conversion_male)


# simple model
model = smf.ols('wage ~ job_experience + age + male + education + job_satisfaction + alc + supervisor + work_time', data=df_tmp)
fit_simple = model.fit()
# print(fit_simple.summary()) # print summary

# simple model with more features and binary education
conversion_education = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1}
df_tmp['education'] = df_tmp['education'].replace(conversion_education)

model = smf.ols('wage ~ job_experience + age + male + education + job_satisfaction + job + alc + supervisor + work_time', data=df_tmp)
fit_simple_2 = model.fit()
# print(fit_simple_2.summary())

# robust model
df_tmp["wage_log"] = np.log(df_tmp["wage"])
model = smf.ols('wage ~ job_experience + age + male + education + job_satisfaction + job + alc + supervisor + work_time + psu + foreigin_lan', data=df_tmp)
fit_robust = model.fit()
# print(robust.summary())
# print(print(robust.compare_f_test(fit_simple_2)))

# robust model with social networks features
model = smf.ols('wage ~ job_experience + age + male + education + job_satisfaction + job + alc + supervisor + work_time + psu + foreigin_lan + ok + vk + facebook + twitter', data=df_tmp)
fit_soc = model.fit()
# print(fit_soc.summary())
# print(print(fit_soc.compare_f_test(fit_robust)))

# robust model with social networks features and log(wage)
df_tmp["wage_log"] = np.log(df_tmp["wage"])
model = smf.ols('wage_log ~ job_experience + age + male + education + job_satisfaction + job + alc + supervisor + work_time + psu + foreigin_lan + ok + vk + facebook + twitter', data=df_tmp)
fit_log_soc = model.fit()
# print(fit_soc.summary())

# prediction
df_pred = df_tmp

df_pred = pd.DataFrame({
    "job_experience":   [2],
    "age":              [22],
    "male":             1,
    "education":        1,
    "job_satisfaction": 2,
    "job":              [27], # it job
    "alc":              [1],
    "supervisor":       [0],
    "psu":              [1], # live in Moscow
    "work_time":        [8],
    "foreigin_lan":     [0],
    "ok":               [0],
    "vk":               [1],
    "facebook":         [0],
    "twitter":          [0]
})

df_tmp__ = df_tmp.copy()
df_tmp__.loc[4,'job_satisfaction'] = 1
df_tmp__.loc[4,'job'] = 27
df_tmp__.loc[4,'alc'] = 0
df_tmp__.loc[4,'psu'] = 1
df_tmp__.loc[4,'education'] = 1
df_tmp__.loc[4,'male'] = 1
df_tmp__.loc[4,'age'] = 22
df_tmp__.loc[4,'work_time'] = 8
df_tmp__.loc[4,'foreigin_lan'] = 0
df_tmp__.loc[4,'supervisor'] = 0
df_tmp__.loc[4,'ok'] = 1
df_tmp__.loc[4,'vk'] = 1
df_tmp__.loc[4,'twitter'] = 0
df_tmp__.loc[4,'facebook'] = 0

#print(fit_soc.predict(df_tmp__))
#print(fit_soc.get_prediction(df_tmp__).summary_frame())



