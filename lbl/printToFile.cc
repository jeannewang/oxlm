// STL
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <iterator>
#include <cstring>
#include <functional>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <set>
#include <map>
#include <vector>
#include <set>

// Boost
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/random.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>

// Local
#include "lbl/log_bilinear_model.h"
#include "lbl/log_add.h"
#include "corpus/corpus.h"
#include "tree/tree.hh"

static const char *REVISION = "$Rev: 247 $";

// Namespaces
using namespace boost;
using namespace boost::program_options;
using namespace std;
using namespace oxlm;
using namespace Eigen;


struct ltnode
{
  bool operator()(tree<float>::breadth_first_queued_iterator const & a,
	tree<float>::breadth_first_queued_iterator const & b) const
  {
    if ( (*a) < (*b) ) {
			return true;
		}
		return false;
  }
};


typedef vector<WordId> Sentence;
typedef vector<WordId> Corpus;

void learn(const variables_map& vm, const ModelData& config);
double sigmoid(double x);
double log_sigmoid(double x);
double log_one_minus_sigmoid(double x);
double getLogWordProb(const HuffmanLogBiLinearModel& model, VectorReal& prediction_vector, WordId w );
tree<float> createReadInTree(Dict& dict, string filename, bool addInStartEnd=true);
set< tree<float>::breadth_first_queued_iterator, ltnode> getRNodesAboveChildren(const tree<float>& binaryTree);
int getInternalNodeCount(tree<float> & binaryTree);
VectorReal perplexity(const HuffmanLogBiLinearModel& model, const Corpus& test_corpus, int stride);
pair< vector< vector< vector<int> > >, vector< vector< vector<int> > > > getYs(tree<float>& binaryTree);
void print_tree(ofstream& file, const tree<float>& tr, tree<float>::pre_order_iterator it, tree<float>::pre_order_iterator end,Dict& dict);

int main(int argc, char **argv) {
  cout << "Online noise contrastive estimation for log-bilinear models with huffman encoded vocabulary: Copyright 2013 Phil Blunsom, " 
       << REVISION << '\n' << endl;

  ///////////////////////////////////////////////////////////////////////////////////////
  // Command line processing
  variables_map vm; 

  // Command line processing
  options_description cmdline_specific("Command line specific options");
  cmdline_specific.add_options()
    ("help,h", "print help message")
    ("config,c", value<string>(), 
        "config file specifying additional command line options")
    ;
  options_description generic("Allowed options");
  generic.add_options()
    ("input,i", value<string>()->default_value("data.txt"), 
        "corpus of sentences, one per line")
    ("test-set", value<string>(), 
        "corpus of test sentences to be evaluated at each iteration")
		("file-out", value<string>()->default_value("fileout.txt"), "file out")
		("tree-in", value<string>()->default_value("../browncluster/output.txt"), "Use tree from this file")
		("variable", value<string>()->default_value("Q"), "variable types are: Q,R,perplexity")
		("model-in,m", value<string>()->default_value("model"),"initial model")
    ("model-out,o", value<string>()->default_value("model"), 
        "base filename of model output files")
		
    ("iterations", value<int>()->default_value(10), 
        "number of passes through the data")
    ("minibatch-size", value<int>()->default_value(100), 
        "number of sentences per minibatch")
    ("instances", value<int>()->default_value(std::numeric_limits<int>::max()), 
        "training instances per iteration")
    ("order,n", value<int>()->default_value(5), 
        "ngram order")
    ("lambda,r", value<float>()->default_value(0.0), 
        "regularisation strength parameter")
    ("dump-frequency", value<int>()->default_value(0), 
        "dump model every n minibatches.")
    ("word-width", value<int>()->default_value(100), 
        "Width of word representation vectors.")
    ("threads", value<int>()->default_value(1), 
        "number of worker threads.")
    ("test-tokens", value<int>()->default_value(10000), 
        "number of evenly spaced test points tokens evaluate.")
    ("step-size", value<float>()->default_value(1.0), 
        "SGD batch stepsize, it is normalised by the number of minibatches.")
    ("classes", value<int>()->default_value(100), 
        "number of classes for factored output.")
    ("verbose,v", "print perplexity for each sentence (1) or input token (2) ")
    ("randomise", "visit the training tokens in random order")
    ("diagonal-contexts", "Use diagonal context matrices (usually faster).")
    ;
  options_description config_options, cmdline_options;
  config_options.add(generic);
  cmdline_options.add(generic).add(cmdline_specific);

  store(parse_command_line(argc, argv, cmdline_options), vm); 
  if (vm.count("config") > 0) {
    ifstream config(vm["config"].as<string>().c_str());
    store(parse_config_file(config, cmdline_options), vm); 
  }
  notify(vm);
  ///////////////////////////////////////////////////////////////////////////////////////

  if (vm.count("help")) { 
    cout << cmdline_options << "\n"; 
    return 1; 
  }

  ModelData config;
  config.l2_parameter = vm["lambda"].as<float>();
  config.word_representation_size = vm["word-width"].as<int>();
  config.threads = vm["threads"].as<int>();
  config.ngram_order = vm["order"].as<int>();
  config.verbose = vm.count("verbose");
  config.classes = vm["classes"].as<int>();

  cout << "################################" << endl;
  cout << "# Config Summary" << endl;
  cout << "# order = " << vm["order"].as<int>() << endl;
  if (vm.count("model-in"))
    cout << "# model-in = " << vm["model-in"].as<string>() << endl;
  cout << "# model-out = " << vm["model-out"].as<string>() << endl;
  cout << "# input = " << vm["input"].as<string>() << endl;
  cout << "# minibatch-size = " << vm["minibatch-size"].as<int>() << endl;
  cout << "# lambda = " << vm["lambda"].as<float>() << endl;
  cout << "# iterations = " << vm["iterations"].as<int>() << endl;
  cout << "# threads = " << vm["threads"].as<int>() << endl;
  cout << "# classes = " << config.classes << endl;
  cout << "################################" << endl;

  omp_set_num_threads(config.threads);

  learn(vm, config);

  return 0;
}

void learn(const variables_map& vm, const ModelData& config) {
  Corpus training_corpus, test_corpus;
  Dict dict;
  dict.Convert("<s>");
  WordId end_id = dict.Convert("</s>");
	tree<float> binaryTree;

  //////////////////////////////////////////////
  // read the training sentences
  ifstream in(vm["input"].as<string>().c_str());
  string line, token;

  while (getline(in, line)) {
    stringstream line_stream(line);
    while (line_stream >> token) 
      training_corpus.push_back(dict.Convert(token));
    training_corpus.push_back(end_id);
  }
  in.close();
  //////////////////////////////////////////////

	//////////////////////////////////////////////
	// read the test sentences
	bool have_test = vm.count("test-set");
	if (have_test) {
	  ifstream test_in(vm["test-set"].as<string>().c_str());
	  while (getline(test_in, line)) {
	    stringstream line_stream(line);
	    Sentence tokens;
	    while (line_stream >> token) {
	      WordId w = dict.Convert(token, true);
	      if (w < 0) {
	        cerr << token << " " << w << endl;
					w=0;
					//TODO: deal with unknown words
	        //assert(!"Unknown word found in test corpus.");
	      }
	      test_corpus.push_back(w);
	    }
	    test_corpus.push_back(end_id);
	  }
	  test_in.close();
	}
	//////////////////////////////////////////////
  //read in the tree
	if (vm.count("tree-in")){
		binaryTree = createReadInTree(dict, vm["tree-in"].as<string>(),false);
	}
	/////////////////////////////////////////////
	//load in model
	int internalNodeCount = getInternalNodeCount(binaryTree);
	HuffmanLogBiLinearModel model(config, dict, vm.count("diagonal-contexts"),binaryTree,internalNodeCount);

  if (vm.count("model-in")) {
		cerr<<"starting to load model "<<vm["model-in"].as<string>()<<endl;
    std::ifstream f(vm["model-in"].as<string>().c_str());
    boost::archive::text_iarchive ar(f);
    ar >> model;
		cerr<<"done loading in model"<<endl;
  }

	pair< vector< vector< vector<int> > >, vector< vector< vector<int> > > > pairYs = getYs(model.huffmanTree);
	model.ys = pairYs.first;
	model.ysInternalIndex = pairYs.second;
/////////////////////////////////////////////

	ofstream myfile;
	myfile.open (vm["file-out"].as<string>().c_str());
	
	if (vm["variable"].as<string>() == "Q"){
		for(int i=0;i<model.labels();i++){
			myfile <<model.label_str(i)<<" ";
			for (int j=0;j<model.Q.row(i).size();j++){
				myfile<<model.Q(i,j)<<" ";
			}
			myfile<<endl;
		}
	}
	else if (vm["variable"].as<string>() == "R"){
		set< tree<float>::breadth_first_queued_iterator, ltnode> RNodes = getRNodesAboveChildren(binaryTree);
		set< tree<float>::breadth_first_queued_iterator, ltnode>::iterator setIt;
		for (setIt=RNodes.begin(); setIt!=RNodes.end(); ++setIt){
			tree<float>::breadth_first_queued_iterator it =(*setIt);
			int numLeafChildren=0;
			for (int i=0;i<binaryTree.number_of_children(it);i++){
				tree<float>::breadth_first_queued_iterator itChild = binaryTree.child(it, i);
				if (binaryTree.isLeaf(itChild)){
					numLeafChildren++;
				}
			}
			int countLeaf=0;
			for (int i=0;i<binaryTree.number_of_children(it);i++){
				tree<float>::breadth_first_queued_iterator itChild = binaryTree.child(it, i);
				if (binaryTree.isLeaf(itChild)){
					myfile<<model.label_str(*itChild);
					if (countLeaf< numLeafChildren-1){
						myfile<<"-";
						countLeaf++;
					}
				}	
			}
			myfile<<" ";
			for (int j=0;j<model.R.row(*it).size();j++){
				myfile<<model.R((*it),j)<<" ";
			}
			myfile<<endl;	
		}	
	}
	else if (vm["variable"].as<string>() == "perplexity"){
		VectorReal perplex_vector = perplexity(model, test_corpus, 1);
		int wordDepth=0;
		for(int i=0;i<model.labels();i++){
			//get wordDepth
			tree<float>::breadth_first_queued_iterator it=binaryTree.begin_breadth_first();
			while(it!=binaryTree.end_breadth_first() && binaryTree.is_valid(it)) {
				if (binaryTree.isLeaf(it) && (*it)==i){
					wordDepth=binaryTree.depth(it);
					break;
				}
				++it;
			}
			//print out: word perplexity wordDepth
			myfile <<model.label_str(i)<<" "<<perplex_vector(i)<<" "<<wordDepth<<endl;
		}
	}
	else if (vm["variable"].as<string>() == "tree"){
		print_tree(myfile, binaryTree, binaryTree.begin(), binaryTree.end(),dict);
	}
	
	myfile.close();
	cerr<<"done writing to "<<vm["file-out"].as<string>()<<endl;

}

void print_tree(ofstream& file, const tree<float>& tr, tree<float>::pre_order_iterator it, tree<float>::pre_order_iterator end,Dict& dict)
	{
	if(!tr.is_valid(it)) return;
	int rootdepth=tr.depth(it);
	file << "-----" << endl;
	while(it!=end) {
		for(int i=0; i<tr.depth(it)-rootdepth; ++i) 
			file << "  ";
		if (tr.isLeaf(it)){
			int index=(*it);
			file << "<" <<index<<"> "<< dict.Convert(index) << endl << flush;
		}
		else{
			file << (*it) << endl << flush;
		}
		++it;
		}
	file << "-----" << endl;
	}
int getInternalNodeCount(tree<float> & binaryTree){
	int internalCount=0;
	tree<float>::breadth_first_queued_iterator it=binaryTree.begin_breadth_first();
	while(it!=binaryTree.end_breadth_first() && binaryTree.is_valid(it)) {
		if (!binaryTree.isLeaf(it)){
			internalCount++;
		}
		++it;
	}
	return internalCount;
}

tree<float> createReadInTree(Dict& dict, string filename, bool addInStartEnd){
	
	tree<float> binaryTree;
	binaryTree.set_head(0);
	
	int tokenCount=0;
	cerr<<"filename:"<<filename<<endl;
	std::ifstream in(filename.c_str());
	string line, token;
	while (getline(in, line)) {
		stringstream line_stream(line);
		line_stream >> token;
		string code = token;
		line_stream >> token;
		string word = token;
		line_stream >> token;
		string freqstr = token;
		int freq=atoi(freqstr.c_str());
		tokenCount+=freq;
		cerr<<"code:"<<code<<" word:"<<word<<" freq:"<<freq<<endl;
		
		tree<float>::pre_order_iterator currentNode=binaryTree.begin();
		currentNode=binaryTree.replace (currentNode, (*currentNode)+freq);
		for (int i=0;i<code.length();i++){
			int y=(code[i]=='0' ? 0 : 1);
			int numChildren=binaryTree.number_of_children(currentNode);
			if (numChildren==0){
				if (y==0){
					binaryTree.append_child(currentNode,freq);
				}
				if (y==1){
					binaryTree.append_child(currentNode,0);
					binaryTree.append_child(currentNode,0);
				}
			}
			else if (numChildren==1){
				if (y==1){
					binaryTree.append_child(currentNode,0);
				}
			}
			currentNode=binaryTree.child(currentNode, y);
			currentNode=binaryTree.replace (currentNode, (*currentNode)+freq);
			
			//at end of path, add node
			if (i==(code.length()-1)){ 
				currentNode=binaryTree.replace (currentNode, dict.Lookup(word));
			}	
		}
	}
	in.close();
	
	int internalCount=0;
	tree<float>::breadth_first_queued_iterator it=binaryTree.begin_breadth_first();
	while(it!=binaryTree.end_breadth_first() && binaryTree.is_valid(it)) {
		if (!binaryTree.isLeaf(it)){
			it=binaryTree.replace (it, internalCount);
			internalCount++;
		}
		++it;
	}
	
	if(addInStartEnd){
			bool noStartEnd=true;
			tree<float>::breadth_first_queued_iterator it=binaryTree.begin_breadth_first();
			int treeDepth=binaryTree.depth(it);
			while(it!=binaryTree.end_breadth_first() && binaryTree.is_valid(it)) {

				if (binaryTree.isLeaf(it)){
					//add in start and end symbol after a random word about 10% down
					if ( noStartEnd && ((float)binaryTree.depth(it)/treeDepth) >= .1 ){
						int word=(*it);
						tree<float>::breadth_first_queued_iterator it1=binaryTree.append_child(it,word);
						tree<float>::breadth_first_queued_iterator it2=binaryTree.append_child(it,internalCount+1);
						it1=binaryTree.replace (it, internalCount);
						binaryTree.append_child(it2,dict.Lookup("<s>"));
						binaryTree.append_child(it2,dict.Lookup("</s>"));
						noStartEnd=false;
					}
				}
				++it;
			}
		}

	return binaryTree;
}

set< tree<float>::breadth_first_queued_iterator, ltnode> getRNodesAboveChildren(const tree<float>& binaryTree){
	
	set< tree<float>::breadth_first_queued_iterator,ltnode > RNodes;
	
	tree<float>::breadth_first_queued_iterator it=binaryTree.begin_breadth_first();
	while(it!=binaryTree.end_breadth_first() && binaryTree.is_valid(it)) {

		if (binaryTree.isLeaf(it)){
			RNodes.insert(binaryTree.parent(it));
		}
		++it;
	}
	return RNodes;
}

VectorReal perplexity(const HuffmanLogBiLinearModel& model, const Corpus& test_corpus, int stride) {

  int word_width = model.config.word_representation_size;
  int context_width = model.config.ngram_order-1;

  int tokens=0;
  WordId start_id = model.label_set().Lookup("<s>");
  WordId end_id = model.label_set().Lookup("</s>");

	//fill context vectors
  vector<MatrixReal> context_vectors(context_width, MatrixReal::Zero(test_corpus.size(), word_width)); 
  for (int instance=0; instance < test_corpus.size(); ++instance) {
    const int& t = instance;
    int context_start = t - context_width;
    bool sentence_start = (t==0);
    for (int i=context_width-1; i>=0; --i) {
      int j=context_start+i;
      sentence_start = (sentence_start || j<0 || test_corpus.at(j) == end_id);
      int v_i = (sentence_start ? start_id : test_corpus.at(j));
      context_vectors.at(i).row(instance) = model.Q.row(v_i);
    }
  }

	//create prediction vectors
  MatrixReal prediction_vectors = MatrixReal::Zero(test_corpus.size(), word_width);
  for (int i=0; i<context_width; ++i)
    prediction_vectors += model.context_product(i, context_vectors.at(i));
	
	
	VectorReal perplex_vector = VectorReal::Zero(model.labels());
	VectorReal normalizer_vector = VectorReal::Zero(model.labels());
  {
    #pragma omp master
    cout << "Calculating perplexity for " << test_corpus.size()/stride << " tokens"<<endl;
	
    size_t thread_num = omp_get_thread_num();
    size_t num_threads = omp_get_num_threads();
		VectorReal prediction_vector;
    for (size_t s = (thread_num*stride); s < test_corpus.size(); s += (num_threads*stride)) {
      WordId w = test_corpus.at(s);
			
			//get log of word probability
			prediction_vector = prediction_vectors.row(s);
			double log_word_prob = getLogWordProb(model,prediction_vector,w);
			//cout<<"word prob "<<model.label_str(test_corpus.at(s))<<": "<<exp(log_word_prob)<<endl;
			
 			perplex_vector(w) += log_word_prob; //multiplying in log space
			normalizer_vector(w)+=1;
  
      tokens++;
    }
    #pragma omp master
    cout << endl;
  }
	for (int w=0;w<model.labels();w++){
		if (normalizer_vector(w)==0){
			normalizer_vector(w)=1;
		}
		perplex_vector(w)=exp(-perplex_vector(w)/normalizer_vector(w));
	}
  return perplex_vector; 
}

double getLogWordProb(const HuffmanLogBiLinearModel& model, VectorReal& prediction_vector, WordId w ){
	
	double log_word_prob = 0;
	for (int winstances=0;winstances<model.ys[w].size();winstances++){
		for (int i=model.ys[w][winstances].size()-2; i>=0;i--){
		
			int y=model.ys[w][winstances][i];
			int yIndex=model.ysInternalIndex[w][winstances][i+1]; //get parent node
			Real wcs = (model.R.row(yIndex) * prediction_vector.matrix()) + model.B(yIndex);
			double binary_conditional_prob = ((y==1) ? log_sigmoid(wcs) : log_one_minus_sigmoid(wcs) ) ;		
			log_word_prob += binary_conditional_prob; //multiplying in log space	
			
		}
	}
  assert(isfinite(log_word_prob));
	return log_word_prob;
	
}

pair< vector< vector< vector<int> > >, vector< vector< vector<int> > > > getYs(tree<float>& binaryTree){
		
		int leafCount=0;
		{
			tree<float>::leaf_iterator it=binaryTree.begin_leaf();
			while(it!=binaryTree.end_leaf() && binaryTree.is_valid(it)) {
				leafCount++;
					++it;
			}
		}
		//store y's in vector of vectors
		vector< vector< vector<int> > > ys(leafCount); //one y vector per word
		vector< vector < vector<int> > >internalIndex(leafCount); //one internal index vector per word
		tree<float>::leaf_iterator itLeaf=binaryTree.begin_leaf();
		while(itLeaf!=binaryTree.end_leaf() && binaryTree.is_valid(itLeaf)) {
			
				//figure out y's for this word
				int wordIndex=(*itLeaf);
				tree<float>::leaf_iterator it;
				it=itLeaf;
				vector<int> ys_wordInstance;
				vector<int> internalIndex_wordInstance;
				while(it!=NULL && it!=binaryTree.end() && binaryTree.is_valid(it)) {
					int y=binaryTree.index(it);
					ys_wordInstance.push_back(y); //order is from the leaf to the root
					int nodeIndex=(int)(*it);
					internalIndex_wordInstance.push_back(nodeIndex);
					//cerr<<(*it)<<" "<<y<<endl;
					it=tree<float>::parent(it);
					
				}
				ys[wordIndex].push_back(ys_wordInstance);
				internalIndex[wordIndex].push_back(internalIndex_wordInstance);
				++itLeaf;
		}
		pair< vector< vector< vector<int> > >, vector< vector< vector<int> > > > returnValue;
		returnValue.first=ys;
		returnValue.second=internalIndex;
		return returnValue;
}

double sigmoid(double x){
	if (x > 0)
		return 1.0/(1+exp(-x));
		
	return 1.0-(1.0/(1+exp(x)));
}

double log_sigmoid(double x) {
	//log(sigmoid(x))
  if (x > 0)
    return - log (1 + exp(-x));

  return x - log (1 + exp(x));
}

double log_one_minus_sigmoid(double x){
	//log(1-sigmoid(x))
	if (x > 0)
		return -x - log (1 + exp(-x));

  return - log (1 + exp(x));
}
