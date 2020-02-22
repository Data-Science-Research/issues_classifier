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
      #Resgatará a classificação das categorias
      model = classifiermodule.returnprediction(request.form['autocomplete'])

      #Resgatará Issues ID similares
      paragraphvector = classifiermodule.returnsimilarparagraph(request.form['autocomplete'])

      #Resgatará troca de mensagens dos Issues similares
      similarissues = classifiermodule.returnissues([paragraphvector[0][0], paragraphvector[1][0], paragraphvector[2][0]])

      return render_template('resultsform.html', problema=request.form['autocomplete'], groups=model, issuesid=paragraphvector, issues=similarissues)

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