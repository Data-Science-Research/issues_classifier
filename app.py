from flask import Flask, render_template, request, jsonify
import classifiermodule
import sys
import itertools

app = Flask('issues_classifier')

@app.route('/')
def show_predict_stock_form():
    return render_template('predictorform.html')

@app.route('/results', methods=['POST'])
def results():
    form = request.form
    if request.method == 'POST':
      #write your function that loads the model
      model = classifiermodule.returnprediction(request.form['year']) #get_model() #you can use pickle to load the trained model
      year = request.form['year']
      predicted_stock_price = 'modelo' #model.predict(year)

      palavra = request.form['palavra']
      synonums = classifiermodule.returnsynonums(request.form['palavra'])

      paragraphvector = classifiermodule.returnsimilarparagraph(request.form['year'])

      similarissues = classifiermodule.returnissues('name')

      return render_template('resultsform.html', year=year, predicted_price=model, palavra=palavra, sinonimo=paragraphvector, issues=similarissues)

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    search = request.args.get('q').split().pop()
    try:
      returnsynonums = classifiermodule.returnsynonums(search)    
      results = [mv for mv in returnsynonums]          
    except Exception as e:
      print('Não encontrou sinônimo!', file=sys.stderr)    
      results = []          
    finally:      
      return jsonify(matching_results=results)
    

app.run("localhost", "9999", debug=True)