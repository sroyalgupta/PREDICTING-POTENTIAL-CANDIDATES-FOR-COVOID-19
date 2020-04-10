from flask import Flask,render_template,request
import pickle

app = Flask(__name__)

file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close()

@app.route('/',methods=['GET','POST'])
def hello_world():
    if request.method == 'POST':
        mydict = request.form
        fever = int(mydict['fever'])
        age = int(mydict['age'])
        bodypain = int(mydict['bodypain'])
        runnynose = int(mydict['runnynose'])
        diffbreath = int(mydict['diffbreath'])
        prob = clf.predict_proba([[fever,bodypain,age,runnynose,diffbreath]])
        return render_template('show.html',inf=prob[0][0])
    return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True)