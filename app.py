from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import datetime
import calendar
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import base64
from pandas.plotting import table
from modelpy import predict_trend , beta
import warnings
warnings.filterwarnings('ignore')



# day_shift and week_shift are same as in modelpy.ipynb
day_shift_d = {
    'index': [0, 1, 2, 3, 4, 5, 6],
    'weightage': [89690.116984, 92449.103997, 25297.310490, 17427.323423, -80149.485614, -59329.172627, -85385.196653],
    'Day': ['Friday', 'Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday']
}
day_shift = pd.DataFrame(day_shift_d)
week_shift_d = {
    'Week': ['00', '01', '02', '03'],
    'seasonal': [7172.315757, -20266.402993, 4879.856302, 8214.230935]
}
week_shift = pd.DataFrame(week_shift_d)
def get_month_start_end(month_name, year=2022):
    if year is None:
        year = datetime.datetime.now().year
    month_number = datetime.datetime.strptime(month_name, '%B').month
    start_date = datetime.date(year, month_number, 1)
    _, last_day = calendar.monthrange(year, month_number)
    end_date = datetime.date(year, month_number, last_day)
    return start_date, end_date

def get_trend_values(day,week,month):
  # for detailed explanation and working of this code please refer to the same function in modelpy.ipynb
  pred_trend=predict_trend(np.array([int(week)+2+52]),beta)
  day_sea = day_shift.loc[day_shift['Day'].values ==day]['weightage'].values;
  week_sea = week_shift.loc[week_shift['Week'].values == str('0'+str(int(week)%4))]['seasonal'].values;
  min_final = pred_trend+min(day_sea,week_sea) +100000
  max_final = pred_trend+max(day_sea,week_sea) -100000
  final_trend = (min_final+max_final)/2
  max_range = abs(min_final-max_final)*100/pred_trend
  return int(min_final),int(max_final),int(final_trend),float(max_range)
def predictions(start_date,end_date,month):
         # for detailed explanation and working of this code please refer modelpy.ipynb
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        start_week = start_date.strftime('%W')
        end_week =end_date.strftime('%W')
        givenmonth = start_date.strftime('%m')
        yearstart = pd.to_datetime('2022-01-01')
        plot_start_day=(start_date-yearstart).days
        plot_end_day = (end_date-yearstart).days
        outputs = []
        for i in range(int(start_week),int(end_week)):
            for k in ('Sunday','Monday','Tuesday','Wednesday','Thursday','Friday',"Saturday"):
                output = get_trend_values(k,str(i),givenmonth)
                outputs.append(output)
        final_array = np.concatenate(outputs)
        change = [inner_list[3] for inner_list in outputs]
        low_list = [inner_list[0] for inner_list in outputs]
        top_list = [inner_list[1] for inner_list in outputs]
        trend_list = [inner_list[2] for inner_list in outputs]
        plt.figure()
        plt.plot(np.arange(plot_start_day,plot_start_day+len(outputs)),[row[0] for row in outputs], label='max')
        plt.plot(np.arange(plot_start_day,plot_start_day+len(outputs)),[row[1] for row in outputs], label='min')
        plt.plot(np.arange(plot_start_day,plot_start_day+len(outputs)),[row[2] for row in outputs], label='trend')
        plt.ylabel('Receipt count in multiples of 10 millions')
        if month == None:
            label_text = 'Receipt count estimate form '+start_date.strftime('%Y-%m-%d')+' to '+end_date.strftime('%Y-%m-%d')
        else:
            label_text = 'Receipt count estimate for ' + month + ' 2022'
        plt.xlabel(label_text)
        plt.xticks([])
        plt.legend()
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode() # image url for HTML
        plt.close() 
        #statistics of our Predicted Values
        stat = pd.DataFrame({
            'stats': ['Minimum(Trend)', 
                      'Maximum(Trend)', 
                      'Max % change',
                      'Maximum ( day and week bias)',
                      'Minimum (day and week bias)',
                      'Mean RC in the duration',
                      'standard deviation'],
             'Stats for given Input': [min(trend_list), max(trend_list), max(change),max(max(top_list),max(low_list)),min(min(low_list),min(top_list)),np.mean(trend_list),np.std(trend_list)]
            })
        stat.set_index('stats', inplace=True)
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.axis('off')
        tbl = table(ax, stat, loc='center')
        img2 = io.BytesIO()
        plt.savefig(img2, format='png', bbox_inches='tight')
        img2.seek(0)
        plot_url2 = base64.b64encode(img2.getvalue()).decode()    #image url for HTML
        plt.close()
        return plot_url,plot_url2
app = Flask(__name__)
model = pickle.load(open('trend.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('myhtmlpage.html')

@app.route('/predict',methods=['POST'])
def predict():
    output = [str(x) for x in request.form.values()]
    month1 =output[0]
    if output[0] != 'Custom Dates':
        print('given month is : ',output[0])
        start_date1, end_date1 = get_month_start_end(output[0])
        plot_url,plot_url2 = predictions(start_date1,end_date1,month1)
        errors = 'no errors'
    elif output[1]== '' or output[2]== '':
        errors = 'Required inputs are not given'
        plot_url = None
        plot_url2= None
    else :
        start_date_str = output[1]
        end_date_str = output[2]
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
        start_week = start_date.strftime('%W')
        end_week =end_date.strftime('%W')
        givenmonth = start_date.strftime('%m')
        if int(end_week) < int(start_week):
            errors = 'dates are not in chronological order'
            plot_url =None
            plot_url2= None
        else:
            plot_url,plot_url2 = predictions(start_date_str,end_date_str,None)
            errors = 'no errors'

    return render_template('myhtmlpage.html', plot_url=plot_url,plot_url2=plot_url2,errors=errors)


if __name__ == "__main__":
    app.run(debug=True)