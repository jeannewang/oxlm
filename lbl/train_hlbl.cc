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
#include <Eigen/SVD>
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

const double pi = 3.141592653589793238462;

typedef vector<WordId> Sentence;
typedef vector<WordId> Corpus;

void print_tree(const tree<float>& tr, tree<float>::pre_order_iterator it, tree<float>::pre_order_iterator end,Dict& dict);
pair< vector< vector< vector<int> > >, vector< vector< vector<int> > > > getYs(tree<float>& binaryTree);
double sigmoid(double x);
double log_sigmoid(double x);
double log_one_minus_sigmoid(double x);
double getLogWordProb(const HuffmanLogBiLinearModel& model, VectorReal& prediction_vector, WordId w );
void highestProbability(const HuffmanLogBiLinearModel& model, const Corpus& test_corpus);
double gaussianCalc(VectorReal& x, VectorReal& mean, MatrixReal& covar);
MatrixReal gaussianMixtureModel(int word_width,int numWords, MatrixReal& expected_prediction_vectors,VectorReal& allwords, bool regularize);
void recursiveAdaptiveHelper(tree<float>& binaryTree, tree<float>::pre_order_iterator oldNode, VectorReal& allwords,int word_width,MatrixReal& expected_prediction_vectors,double epsilon);
double getUnigramForNode(HuffmanLogBiLinearModel& model, tree<float>& binaryTree, tree<float>::pre_order_iterator node);
int getInternalNodeCount(tree<float> & binaryTree);

void learn(const variables_map& vm, const ModelData& config);

typedef int TrainingInstance;
typedef vector<TrainingInstance> TrainingInstances;
void cache_data(int start, int end, 
                const Corpus& training_corpus, 
                const vector<size_t>& indices,
                TrainingInstances &result);

Real sgd_gradient(HuffmanLogBiLinearModel& model,
               const Corpus& training_corpus,
               const TrainingInstances &training_instances,
               Real lambda, 
               LogBiLinearModel::WordVectorsType& g_R,
               LogBiLinearModel::WordVectorsType& g_Q,
               LogBiLinearModel::ContextTransformsType& g_C,
               LogBiLinearModel::WeightsType& g_B);



Real perplexity(const HuffmanLogBiLinearModel& model, const Corpus& test_corpus, int stride=1);

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
    ("iterations", value<int>()->default_value(10), 
        "number of passes through the data")
    ("minibatch-size", value<int>()->default_value(100), 
        "number of sentences per minibatch")
    ("instances", value<int>()->default_value(std::numeric_limits<int>::max()), 
        "training instances per iteration")
    ("order,n", value<int>()->default_value(3), 
        "ngram order")
    ("model-in,m", value<string>(), 
        "initial model")
    ("model-out,o", value<string>()->default_value("model"), 
        "base filename of model output files")
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
		("tree-in", value<string>()->default_value("../browncluster/output.txt"), "Use tree from this file")
		("tree-type", value<string>()->default_value("random"), "Tree-types are: random, browncluster, adaptive, huffman")
		("random-model-in", value<string>()->default_value("model"), "Archive of random-tree hlbl model")
		("tree-out", value<string>()->default_value("tree"), "Write tree to file")
		("epsilon", value<double>()->default_value(0), "Epsilon for adaptive epsilon tree")
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
	cout << "# tree-out = " << vm["tree-out"].as<string>() << endl;
  cout << "# input = " << vm["input"].as<string>() << endl;
	cout << "# tree-in = " << vm["tree-in"].as<string>() << endl;
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
void write_out_tree(HuffmanLogBiLinearModel& model, string filename, int trainingSize){
	//open file
	std::ofstream f(filename.c_str());
	
	tree<float> binaryTree = model.huffmanTree;
	tree<float>::leaf_iterator itLeaf=binaryTree.begin_leaf();
	while(itLeaf!=binaryTree.end_leaf() && binaryTree.is_valid(itLeaf)) {
		
			//figure out y's for this word
			int wordIndex=(*itLeaf);
			for (int winstances=0;winstances<model.ys[wordIndex].size();winstances++){
				for (int i=model.ys[wordIndex][winstances].size()-2;i>=0;i--){
					f<<model.ys[wordIndex][winstances][i];
				}
				f<<'\t'<<model.label_str(wordIndex)<<'\t'<<model.unigram(wordIndex)/model.ys[wordIndex].size()*trainingSize<<endl;
			}
			++itLeaf;
	}
	
	f.close();
}
void print_tree(const tree<float>& tr, tree<float>::pre_order_iterator it, tree<float>::pre_order_iterator end,Dict& dict)
	{
	if(!tr.is_valid(it)) return;
	int rootdepth=tr.depth(it);
	cerr << "-----" << endl;
	while(it!=end) {
		for(int i=0; i<tr.depth(it)-rootdepth; ++i) 
			cerr << "  ";
		if (tr.isLeaf(it)){
			int index=(*it);
			cerr << "<" <<index<<"> "<< dict.Convert(index) << endl << flush;
		}
		else{
			cerr << (*it) << endl << flush;
		}
		++it;
		}
	cerr << "-----" << endl;
	}

tree<float> createHuffmanTree(VectorReal& unigram, Dict& dict){

	multimap<float, tree<float> , less<float> > priQ;
	tree<float> huffmanTree;
	//create huffman tree using unigram freq
   for (size_t i=0; i<dict.size(); i++) {
		//make new tree node
		tree<float> node;
		node.set_head(i); // key is index into vocabulary 
		priQ.insert( pair<float,tree<float> >( unigram(i), node )); //lowest probability in front
   }
	while(priQ.size() >1){
		//Get the two nodes of highest priority (lowest frequency) from the queue
		multimap< float,tree<float> >::iterator it1 = priQ.begin();
		multimap< float,tree<float> >::iterator it2 = it1++;
		//Create a new internal node with these two nodes as children and with probability equal to the sum of the two nodes' probabilities.
		//Add the new node to the queue.
		float priority=(*it1).first+(*it2).first;
		tree<float> node;
		node.set_head(priority);
		tree<float> t1=(*it1).second;
		tree<float> t2=(*it2).second;
		node.append_children(node.begin(),t1.begin(),t1.end());
		node.append_children(node.begin(),t2.begin(),t2.end());
		priQ.insert( pair<float,tree<float> >( priority, node ));
		//Remove the two nodes of highest priority (lowest probability) from the queue
		priQ.erase(it1);
		priQ.erase(it2);
	}
	cerr<<"finished priQ"<<endl;
	//The remaining node is the root node and the tree is complete.
	huffmanTree=(*priQ.begin()).second;
	return huffmanTree;
}
	
tree<float> createRandomTree(Dict& dict){

	boost::mt19937 gen;
	
	multimap<float, tree<float> > priQ;
	tree<float> binaryTree;
	//create random tree
  for (size_t i=0; i<dict.size(); i++) {
		//make new tree node
		tree<float> node;
		node.set_head(i); // key is index into vocabulary 
		priQ.insert( pair<float,tree<float> >( 0, node ));
   }
	while(priQ.size() >1){
		//Get two random nodes from the queue
		boost::uniform_int<> dist(0, priQ.size()-1);
		boost::variate_generator<boost::mt19937&, boost::uniform_int<> > die(gen, dist);
		int rand1=die();
		int rand2=die();
		while (rand1 == rand2){
			rand2=die();
		}
		multimap< float,tree<float> >::iterator it1 = priQ.begin();
		multimap< float,tree<float> >::iterator it2 = priQ.begin();
		for (int i=0;i<rand1;i++){
			it1++;
		}
		for (int i=0;i<rand2;i++){
			it2++;
		}
		//Create a new internal node with these two nodes as children and with probability equal to the sum of the two nodes' probabilities.
		//Add the new node to the queue.
		float priority=(*it1).first+(*it2).first;
		tree<float> node;
		node.set_head(priority);
		tree<float> t1=(*it1).second;
		tree<float> t2=(*it2).second;
		node.append_children(node.begin(),t1.begin(),t1.end());
		node.append_children(node.begin(),t2.begin(),t2.end());
		priQ.insert( pair<float,tree<float> >( priority, node ));
		//Remove the two random nodes from the queue
		priQ.erase(it1);
		priQ.erase(it2);
	}
	cerr<<"finished priQ"<<endl;
	//The remaining node is the root node and the tree is complete.
	binaryTree=(*priQ.begin()).second;
	
	return binaryTree;
}

tree<float> createReadInTree(Dict& dict, string filename, bool addInStartEnd=true){
	
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
	
	int internalCount = getInternalNodeCount(binaryTree);
	
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

tree<float> createAdaptiveTree(HuffmanLogBiLinearModel& randModel,Corpus& test_corpus,double epsilon=0){
	Dict dict = randModel.label_set();
	int numWords =dict.size();
//	assert(dict == model.label_set()); //random model must be of the same dataset
	
	tree<float> binaryTree;
	binaryTree.set_head(-1);
	
	int word_width = randModel.config.word_representation_size;
  int context_width = randModel.config.ngram_order-1;

  int tokens=0;
  WordId start_id = randModel.label_set().Lookup("<s>");
  WordId end_id = randModel.label_set().Lookup("</s>");

	//fill context vectors
  vector<MatrixReal> context_vectors(context_width, MatrixReal::Zero(test_corpus.size(), word_width)); 
  for (int instance=0; instance < test_corpus.size(); ++instance) {
    const TrainingInstance& t = instance;
    int context_start = t - context_width;
    bool sentence_start = (t==0);
    for (int i=context_width-1; i>=0; --i) {
      int j=context_start+i;
      sentence_start = (sentence_start || j<0 || test_corpus.at(j) == end_id);
      int v_i = (sentence_start ? start_id : test_corpus.at(j));
      context_vectors.at(i).row(instance) = randModel.Q.row(v_i);
    }
  }

	//create prediction vectors
  MatrixReal prediction_vectors = MatrixReal::Zero(test_corpus.size(), word_width);
  for (int i=0; i<context_width; ++i)
    prediction_vectors += randModel.context_product(i, context_vectors.at(i));

	MatrixReal expected_prediction_vectors = MatrixReal::Zero(numWords, word_width);
	VectorReal epv_count = VectorReal::Zero(numWords);
	for (int instance=0; instance < test_corpus.size(); ++instance) {
		WordId w = test_corpus.at(instance);
		expected_prediction_vectors.row(w) += prediction_vectors.row(instance);
		epv_count(w) += 1;
	}
	for(int w=0;w<numWords;w++){
		if (epv_count(w)==0){
			std::random_device rd;
			std::mt19937 gen(rd());
		  std::normal_distribution<Real> gaussian(0,0.1);
			for (int k=0; k<word_width; k++){
				expected_prediction_vectors(w,k) = gaussian(gen);
			}
		}
		else{
			expected_prediction_vectors.row(w) /= epv_count(w);
		}
	}
	
	//recursively apply mixture of gaussian clustering into 2 clusters.
	VectorReal allwords = VectorReal::Zero(numWords);
	for (int i=0;i<numWords;i++){
		allwords(i)=i;
	}
	recursiveAdaptiveHelper(binaryTree, binaryTree.begin(), allwords,word_width,expected_prediction_vectors,epsilon);
	
	binaryTree=binaryTree.child(binaryTree.begin(), 0); //because there is one extra node
	
	{
		//getting rid of superfluous nodes
		tree<float>::breadth_first_queued_iterator it=binaryTree.begin_breadth_first();
		while(it!=binaryTree.end_breadth_first() && binaryTree.is_valid(it)) {
			
			if (!binaryTree.isLeaf(it)){
				if (binaryTree.number_of_children(it)==1 && !binaryTree.isLeaf(binaryTree.child(it,0)) ){
				//can delete
					cerr<<"deleting"<<endl;
					tree<float>::breadth_first_queued_iterator c1=binaryTree.child(it,0);
					
					c1=binaryTree.insert_subtree_after(it, c1);
					binaryTree.erase_children(it);
					binaryTree.erase(it);
					it=c1;
				}
			}				
			++it;
		}
	}
	
	return binaryTree;
}

double getUnigramForNode(HuffmanLogBiLinearModel& model,tree<float>& binaryTree, tree<float>::pre_order_iterator node){
	if (binaryTree.isLeaf(node)){
		int wordInstances=model.ys[(*node)].size();
		return model.unigram((*node))/wordInstances;
	}
	else{
		double result=0;
		for(int i=0;i<binaryTree.number_of_children(node);i++){
			result+=getUnigramForNode(model,binaryTree,binaryTree.child(node,i));
		}
		return result;
	}
}

void recursiveAdaptiveHelper(tree<float>& binaryTree, tree<float>::pre_order_iterator oldNode, VectorReal& allwords,int word_width,MatrixReal& expected_prediction_vectors,double epsilon){
	
	int numWords=allwords.size();
	if (numWords==0){
		return;
	}
	
	int numChildren=binaryTree.number_of_children(oldNode);
	if (numWords==1){
		if (numChildren <= 1){
			binaryTree.append_child(oldNode,allwords(0));
			cerr<<__LINE__<<":"<<allwords(0)<<endl;
		}
		else{
			tree<float>::breadth_first_queued_iterator c1=binaryTree.child(oldNode,0);
			tree<float>::breadth_first_queued_iterator cNew=binaryTree.append_child(oldNode,-1);
			tree<float>::breadth_first_queued_iterator subtree1=binaryTree.append_child(cNew,-1); //this needs to be a subtree
			binaryTree.replace(subtree1,c1);
			binaryTree.append_child(cNew,allwords(0));
			binaryTree.erase_children(c1);
			binaryTree.erase(c1);
			if (binaryTree.number_of_children(cNew)>2){
				cerr<<"too many children line:"<<__LINE__<<endl;
			}
			cerr<<__LINE__<<":"<<allwords(0)<<endl;
		}
		return;
	}
	else if (numWords==2){
		if(numChildren==0){
			if (oldNode==binaryTree.child(binaryTree.begin(),0)){
				tree<float>::breadth_first_queued_iterator c1=binaryTree.append_child(oldNode,-1);
				binaryTree.append_child(c1,allwords(0)); 
				binaryTree.append_child(c1,allwords(1));
				if (binaryTree.number_of_children(oldNode)>2){
					cerr<<"too many children line:"<<__LINE__<<endl;
				}
				if (binaryTree.number_of_children(c1)>2){
					cerr<<"too many children line:"<<__LINE__<<endl;
				}
			}
			else{
				binaryTree.append_child(oldNode,allwords(0)); 
				binaryTree.append_child(oldNode,allwords(1));
				if (binaryTree.number_of_children(oldNode)>2){
					cerr<<"too many children line:"<<__LINE__<<endl;
				}
			}
			cerr<<__LINE__<<":"<<allwords(0)<<endl;
			cerr<<__LINE__<<":"<<allwords(1)<<endl;
		}
		else if (numChildren==1){
			tree<float>::breadth_first_queued_iterator c2=binaryTree.append_child(oldNode,-1);
			binaryTree.append_child(c2,allwords(0));
			binaryTree.append_child(c2,allwords(1));
			cerr<<__LINE__<<":"<<allwords(0)<<endl;
			cerr<<__LINE__<<":"<<allwords(1)<<endl;
			if (binaryTree.number_of_children(c2)>2){
				cerr<<"too many children line:"<<__LINE__<<endl;
			}
		}
		else{
			tree<float>::breadth_first_queued_iterator c1=binaryTree.child(oldNode,0);
			tree<float>::breadth_first_queued_iterator c2=binaryTree.child(oldNode,1);
			tree<float>::breadth_first_queued_iterator cNew1=binaryTree.append_child(oldNode,-1);
			tree<float>::breadth_first_queued_iterator cNew2=binaryTree.append_child(oldNode,-1);
			
			tree<float>::breadth_first_queued_iterator subtree1=binaryTree.append_child(cNew1,-1);
			binaryTree.replace(subtree1,c1);
			binaryTree.append_child(cNew1,allwords(0));
			binaryTree.erase_children(c1);
			binaryTree.erase(c1);
			if (binaryTree.number_of_children(cNew1)>2){
				cerr<<"too many children line:"<<__LINE__<<endl;
			}

			tree<float>::breadth_first_queued_iterator subtree2=binaryTree.append_child(cNew2,-1);
			binaryTree.replace(subtree2,c2);
			binaryTree.append_child(cNew2,allwords(1));
			binaryTree.erase_children(c2);
			binaryTree.erase(c2);
			if (binaryTree.number_of_children(cNew2)>2){
				cerr<<"too many children line:"<<__LINE__<<endl;
			}
			
			cerr<<__LINE__<<":"<<allwords(0)<<endl;
			cerr<<__LINE__<<":"<<allwords(1)<<endl;
		}
		return;
	}
	else{
		//create new node and attach this to the tree as a child of last node
		tree<float>::pre_order_iterator newNode;
		if (binaryTree.number_of_children(oldNode)>=2){
			tree<float>::breadth_first_queued_iterator c1=binaryTree.child(oldNode,0);
			tree<float>::breadth_first_queued_iterator cNew1=binaryTree.append_child(oldNode,-1);
			tree<float>::breadth_first_queued_iterator subtree1=binaryTree.append_child(cNew1,-1);
			binaryTree.replace(subtree1,c1);
			binaryTree.erase_children(c1);
			binaryTree.erase(c1);
			newNode=cNew1;
		}
		else{
			newNode=binaryTree.append_child(oldNode,-1); 
		}
		MatrixReal respons=gaussianMixtureModel(word_width,numWords,expected_prediction_vectors,allwords,true);
		
		vector<int> group1;
		vector<int> group2;
		for (int i=0;i<numWords;i++){
			if (epsilon == 0){ //do normal adaptive
				if( respons(1,i) >= (1-respons(1,i))){
					group1.push_back(respons(0,i));
				}
				else{
					group2.push_back(respons(0,i));
				}
			}
			else{ //do adaptive(epsilon)
				if ((respons(1,i) >= .5-epsilon) && (respons(1,i) <= .5+epsilon) && ((1-respons(1,i)) >= .5-epsilon) && ((1-respons(1,i)) <= .5+epsilon) ){
					group1.push_back(respons(0,i));
					group2.push_back(respons(0,i));
				}
				else if( respons(1,i) >= (1-respons(1,i))){
					group1.push_back(respons(0,i));
				}
				else{
					group2.push_back(respons(0,i));
				}
			}
		}
		if (group1.size() == numWords && group2.size()!=0){ //not shrinking
				group1.erase(group1.begin());
		}
		if (group2.size() == numWords && group1.size()!=0){ //not shrinking
				group2.pop_back();
		}
		
		if (group1.size() == 0 || group2.size() == 0){
			if (group1.size()==0){
				for (int i=0;i<numWords/2;i++){
					group1.push_back(group2.back());
					group2.pop_back();
				}
			}
			else{
				for (int i=0;i<numWords/2;i++){
					group2.push_back(group1.back());
					group1.pop_back();
				}
			}
		}
		VectorReal group1V=VectorReal::Zero(group1.size());
		VectorReal group2V=VectorReal::Zero(group2.size());
		for (int j=0;j<group1.size();j++){
			group1V(j)=group1[j];
		}
		for (int j=0;j<group2.size();j++){
			group2V(j)=group2[j];
		}
		recursiveAdaptiveHelper(binaryTree, newNode, group1V, word_width,expected_prediction_vectors,epsilon);
		recursiveAdaptiveHelper(binaryTree, newNode, group2V, word_width,expected_prediction_vectors,epsilon);
	}
	
}

double getLoglikelihoodGMM(MatrixReal& expected_prediction_vectors,VectorReal& mixingCoeffs, MatrixReal& means, vector<MatrixReal>& covariances, VectorReal& allwords){
	double loglikelihood=0;
	VectorReal mean0 =means.row(0);
	VectorReal mean1 =means.row(1);
	for (int i=0;i<allwords.size();i++){
		VectorReal x =expected_prediction_vectors.row(allwords(i));
		double mixtureComponent0=mixingCoeffs(0)*gaussianCalc(x,mean0,covariances.at(0));
		double mixtureComponent1=mixingCoeffs(1)*gaussianCalc(x,mean1,covariances.at(1));
		loglikelihood+=log(mixtureComponent0+mixtureComponent1);
	}
	return loglikelihood;
}

MatrixReal gaussianMixtureModel(int word_width,int numWords, MatrixReal& expected_prediction_vectors,VectorReal& allwords, bool regularize=true){
	int numClusters=2;
	int numDim = word_width;
	int numIterations=10; 
	double regularizeEpsilon=.0001;
	MatrixReal means = MatrixReal::Zero(numClusters,numDim);
	vector<MatrixReal> covariances(numClusters, MatrixReal::Zero(numDim, numDim)); 
	VectorReal mixingCoeffs = VectorReal::Zero(numClusters);
	
	/*
	In general, you can avoid getting ill-conditioned covariance matrices by using one of the following precautions:

	Pre-process your data to remove correlated features.
	Set 'SharedCov' to true to use an equal covariance matrix for every component.
	Set 'CovType' to 'diagonal'.
	Use 'Regularize' to add a very small positive number to the diagonal of every covariance matrix.
	Try another set of initial values.
	*/
	
	//initalize
	for (int i=0;i<numWords;i++){
		if (i<(numWords/2)){
			means.row(0) += expected_prediction_vectors.row(allwords(i));
		}
		else{
			means.row(1) += expected_prediction_vectors.row(allwords(i));
		}
	}
	means.row(0)/=(numWords/2);
	means.row(1)/=(numWords-(numWords/2));

	mixingCoeffs(0)=.5;
	mixingCoeffs(1)=.5;
	
	cerr<<"numwords:"<<numWords<<endl;

	MatrixReal M=MatrixReal::Zero(numWords,numDim);
	for (int i=0;i<numWords;i++){
		M.row(i) = expected_prediction_vectors.row(allwords(i));
	}
	for (int i=0; i<numClusters; i++){
		VectorReal ones = VectorReal::Ones(numWords);
		MatrixReal inner=(M-ones*means.row(i));
		covariances.at(i)= (1.0/(numWords-1))* inner.transpose() * inner;
	}
	
	//evaluate the loglikelihood
	double loglikelihood= getLoglikelihoodGMM(expected_prediction_vectors,mixingCoeffs, means, covariances, allwords);

	if (!isfinite(loglikelihood)){
		covariances.at(0).setIdentity();
		covariances.at(1).setIdentity();
		loglikelihood = getLoglikelihoodGMM(expected_prediction_vectors,mixingCoeffs, means, covariances, allwords);
	}

	cerr<<"before update loglikelihood:"<<loglikelihood<<endl;
	
	MatrixReal respons=MatrixReal::Zero(2,allwords.size());
	for (int i=0;i<allwords.size();i++){
		VectorReal x =expected_prediction_vectors.row(allwords(i));
		VectorReal mean0 =means.row(0);
		VectorReal mean1 =means.row(1);
		double mixtureComponent0=mixingCoeffs(0)*gaussianCalc(x,mean0,covariances.at(0));
		double mixtureComponent1=mixingCoeffs(1)*gaussianCalc(x,mean1,covariances.at(1));
		double z =mixtureComponent0+mixtureComponent1;
		respons(0,i)=allwords(i);
		respons(1,i) = mixtureComponent0/z;
	}
	
	int iteration=0;
	bool notConverged=true;
	double lastLogLikelihood=loglikelihood;
	MatrixReal lastrespons=respons;
	while (iteration<numIterations && notConverged){
		//E step: calculate the responsibilities for the current parameter values
		respons=MatrixReal::Zero(2,allwords.size());
		for (int i=0;i<allwords.size();i++){
			VectorReal x =expected_prediction_vectors.row(allwords(i));
			VectorReal mean0 =means.row(0);
			VectorReal mean1 =means.row(1);
			double mixtureComponent0=mixingCoeffs(0)*gaussianCalc(x,mean0,covariances.at(0));
			double mixtureComponent1=mixingCoeffs(1)*gaussianCalc(x,mean1,covariances.at(1));
			double z =mixtureComponent0+mixtureComponent1;
			respons(0,i)=allwords(i);
			respons(1,i) = mixtureComponent0/z;
		}
	
		//M step: reestimate the parameters using the current responsibilities
		double Nk=0;
		for (int i=0;i<allwords.size();i++){
			Nk+=respons(1,i);
		}
		//zero out
		means=MatrixReal::Zero(numClusters,numDim);
		covariances.at(0)=MatrixReal::Zero(numDim, numDim);
		covariances.at(1)=MatrixReal::Zero(numDim, numDim);
	
		//update means
		for (int i=0;i<allwords.size();i++){
			VectorReal x =expected_prediction_vectors.row(allwords(i));
			means.row(0)+=respons(1,i)*x.transpose();
			means.row(1)+=(1-respons(1,i))*x.transpose();
		}
		means.row(0) /=Nk;
		means.row(1) /=(allwords.size()-Nk);
		//cerr<<"N:"<<allwords.size()<<" Nk:"<<Nk<<" N-Nk:"<<allwords.size()-Nk<<endl;
			
		//update covariance matrices
		for (int i=0;i<allwords.size();i++){
			VectorReal x =expected_prediction_vectors.row(allwords(i));
			covariances.at(0) += respons(1,i)*(x-means.row(0).transpose())*(x.transpose()-means.row(0));
			covariances.at(1) += (1-respons(1,i))*(x-means.row(1).transpose())*(x.transpose()-means.row(1));
			if (regularize){
				for (int j=0;j<numClusters;j++){
					for(int k=0;k<numDim;k++){
						covariances.at(j)(k,k)+=regularizeEpsilon;
					}
				}
			}
		}
		covariances.at(0) /= Nk;
		covariances.at(1) /= (allwords.size()-Nk);
		//update mixing coeffs
		mixingCoeffs(0)=Nk/allwords.size();
		mixingCoeffs(1)=(allwords.size()-Nk)/allwords.size();
	
		//evaluate the loglikelihood
		double loglikelihood=getLoglikelihoodGMM(expected_prediction_vectors,mixingCoeffs, means, covariances, allwords);
		cerr<<"loglikelihood:"<<loglikelihood<<endl;
		if (!isfinite(loglikelihood)){
			return lastrespons;
		}
		if (abs(lastLogLikelihood - loglikelihood) < 100){
			notConverged=false;
			//cerr<<"finished in iteration "<<iteration<<endl;
		}
		lastrespons=respons;
		lastLogLikelihood=loglikelihood;
		iteration++;
	}
	return respons;
}

double gaussianCalc(VectorReal& x, VectorReal& mean, MatrixReal& covar){

	MatrixReal covarDiag=covar.diagonal().asDiagonal();
	double determinant=abs(covarDiag.determinant());
	if (determinant==0){ //todo is this underflowing?
		determinant=1;
	}
	MatrixReal invCovar =covarDiag.inverse();
	double innerProd=(x-mean).transpose()*invCovar*(x-mean);
	
	double normalizer = pow(2*pi,x.size()*.5)*pow(determinant,.5);
	double result= (1.0/normalizer)*exp(-.5*innerProd);
	return result;
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

void learn(const variables_map& vm, const ModelData& config) {
  Corpus training_corpus, test_corpus;
  Dict dict;
  dict.Convert("<s>");
  WordId end_id = dict.Convert("</s>");

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
	cerr<<"TEST-SET SIZE:"<<test_corpus.size()<<endl;
  //////////////////////////////////////////////

	//get unigram distribution
  vector<size_t> training_indices(training_corpus.size());
  VectorReal unigram = VectorReal::Zero(dict.size());
  for (size_t i=0; i<training_indices.size(); i++) {
    unigram(training_corpus[i]) += 1;
    training_indices[i] = i;
  }
  unigram /= unigram.sum();
	//////////////////////////////////////////////
	
	//create binarytree from vocabulary
	clock_t tree_start=clock();
	
	tree<float> binaryTree;
	string treeType=vm["tree-type"].as<string>();
	if (treeType == "random") {
		cout <<"Creating Random Tree"<<endl;
		binaryTree = createRandomTree(dict);
		
	}
	else if (treeType == "browncluster") {
			
		if (!vm.count("tree-in")){
			cout<<"Must include option --tree-in"<<endl;
			return;
		}
		cout <<"Creating Brown Cluster Tree with paths from file:"<<vm["tree-in"].as<string>()<<endl;
		binaryTree = createReadInTree(dict, vm["tree-in"].as<string>(),true);
		
	}
	else if (treeType == "adaptive") {
		
		
		if (!vm.count("random-model-in") || !vm.count("tree-in")) {
			cout<<"Must include option --random-model-in and --tree-in"<<endl;
			return;
		}
		tree<float> randomTree = createReadInTree(dict, vm["tree-in"].as<string>(),false);
		int internalNodeCount = getInternalNodeCount(randomTree);
		HuffmanLogBiLinearModel randModel(config, dict, vm.count("diagonal-contexts"),randomTree,internalNodeCount);
		
    std::ifstream f(vm["random-model-in"].as<string>().c_str());
    boost::archive::text_iarchive ar(f);
    ar >> randModel;
	  
		cout<<"Creating Adaptive Tree with random tree from archive file:"<<vm["random-model-in"].as<string>()<<endl;
		if (vm.count("epsilon")){
			binaryTree = createAdaptiveTree(randModel, training_corpus,vm["epsilon"].as<double>());
		}
		else{
			binaryTree = createAdaptiveTree(randModel, training_corpus);
		}
	}
	else if (treeType == "huffman"){
		cout<<"Creating Huffman Tree"<<endl;
		binaryTree=createHuffmanTree(unigram, dict);
	}
	else if (treeType == "huffmanOvercomplete"){
		cout<<"Creating Huffman Tree Overcomplete"<<endl;
		tree<float> subtree1=createHuffmanTree(unigram, dict);
		tree<float> subtree2=createRandomTree(dict);
		tree<float> newTree;
		newTree.set_head(-1);
		tree<float>::breadth_first_queued_iterator c1=newTree.append_child(newTree.begin());
		newTree.insert_subtree(c1,subtree1.begin());
		newTree.insert_subtree(c1,subtree2.begin());
		newTree.erase(c1);
		binaryTree=newTree;		
	}
	else if (treeType != "browncluster" && vm.count("tree-in")){
		binaryTree = createReadInTree(dict, vm["tree-in"].as<string>(),false);	
	}
	
	Real tree_time = (clock()-tree_start) / (Real)CLOCKS_PER_SEC;
	cout<<"Time to create tree:"<<tree_time<<" seconds"<<endl;
	
	////////////////////////////////////////////////////////////////
	
	int internalNodeCount = getInternalNodeCount(binaryTree);
  HuffmanLogBiLinearModel model(config, dict, vm.count("diagonal-contexts"),binaryTree,internalNodeCount);

  if (vm.count("model-in")) {
    std::ifstream f(vm["model-in"].as<string>().c_str());
    boost::archive::text_iarchive ar(f);
    ar >> model;
  }

	model.unigram=unigram;
	
	/////////////////////////////////////////////////////////////////
	
	//set node value to index into R
	int internalCount=0;
	tree<float>::breadth_first_queued_iterator it=model.huffmanTree.begin_breadth_first();
	while(it!=model.huffmanTree.end_breadth_first() && model.huffmanTree.is_valid(it)) {
		if (!model.huffmanTree.isLeaf(it)){
			it=model.huffmanTree.replace (it, internalCount);
			internalCount++;
		}
		++it;
	}
	
	//get binary decisions per word in huffmantree
	pair< vector< vector< vector<int> > >, vector< vector< vector<int> > > > pairYs = getYs(model.huffmanTree);
	model.ys = pairYs.first;
	model.ysInternalIndex = pairYs.second;
	
	//update bias B terms with unigram distribution
	bool updateBWithUnigram=true;
	internalCount=0;
	it=model.huffmanTree.begin_breadth_first();
	while(it!=model.huffmanTree.end_breadth_first() && model.huffmanTree.is_valid(it)) {
		if (!model.huffmanTree.isLeaf(it)){
			 if(updateBWithUnigram){
			 		model.B(internalCount)=getUnigramForNode(model, model.huffmanTree, it); //update with unigram probability
			 	}
			//it=model.huffmanTree.replace (it, model.B(internalCount));
			internalCount++;
		}
		++it;
	}
	
	// for(int w=0;w<model.labels();w++){
	// 	for (int winstances=0;winstances<model.ys[w].size();winstances++){
	// 	cout<<"ysInternalIndex for "<< model.label_str(w) <<" instance "<<winstances<<endl;
	// 		for (int i=model.ys[w][winstances].size()-2; i>=0;i--)
	// 			cout<< model.ysInternalIndex[w][winstances][i+1] <<",";
	// 		cout<<endl;
	// 		for (int i=model.ys[w][winstances].size()-2; i>=0;i--)
	// 			cout<< model.ys[w][winstances][i] <<",";
	// 		cout<<endl;
	// 	}
	// }
	
	print_tree(model.huffmanTree, model.huffmanTree.begin(), model.huffmanTree.end(),dict);
	
	if (vm.count("tree-out")) {
		cout<<"Writing tree to "<<vm["tree-out"].as<string>()<<endl;
		write_out_tree(model, vm["tree-out"].as<string>(), training_corpus.size());
  }

	///////////////////////////////////////////////////////////////////////////////

  VectorReal adaGrad = VectorReal::Zero(model.num_weights());
  VectorReal global_gradient(model.num_weights());
  Real av_f=0.0;
  Real pp=0;

  #pragma omp parallel shared(global_gradient)
  {
    //////////////////////////////////////////////
    // setup the gradient matrices
		int num_nodes = model.internalNodeCount;
    int num_words = model.labels();
    int word_width = model.config.word_representation_size;
    int context_width = model.config.ngram_order-1;

    int R_size = num_nodes*word_width;
    int Q_size = num_words*word_width;
    int C_size = (vm.count("diagonal-contexts") ? word_width : word_width*word_width);
    int B_size = num_nodes;
    int M_size = context_width;
		
    assert((R_size+Q_size+context_width*C_size+B_size+M_size) == model.num_weights());

    Real* gradient_data = new Real[model.num_weights()];
    LogBiLinearModel::WeightsType gradient(gradient_data, model.num_weights());

    LogBiLinearModel::WordVectorsType g_R(gradient_data, num_nodes, word_width);
    LogBiLinearModel::WordVectorsType g_Q(gradient_data+R_size, num_words, word_width);

    LogBiLinearModel::ContextTransformsType g_C;
    Real* ptr = gradient_data+R_size+Q_size;
    for (int i=0; i<context_width; i++) {
      if (vm.count("diagonal-contexts"))
          g_C.push_back(LogBiLinearModel::ContextTransformType(ptr, word_width, 1));
      else
          g_C.push_back(LogBiLinearModel::ContextTransformType(ptr, word_width, word_width));
      ptr += C_size;
    }

    LogBiLinearModel::WeightsType g_B(ptr, B_size);
    LogBiLinearModel::WeightsType g_M(ptr+B_size, M_size);
    //////////////////////////////////////////////

    size_t minibatch_counter=0;
    size_t minibatch_size = vm["minibatch-size"].as<int>();
    for (int iteration=0; iteration < vm["iterations"].as<int>(); ++iteration) {
      clock_t iteration_start=clock();
      #pragma omp master
      {
        av_f=0.0;
        pp=0.0;
        cout << "Iteration " << iteration << ": "; cerr.flush();

        if (vm.count("randomise"))
          std::random_shuffle(training_indices.begin(), training_indices.end());
      }

      TrainingInstances training_instances;
      Real step_size = vm["step-size"].as<float>(); //* minibatch_size / training_corpus.size();

      for (size_t start=0; start < training_corpus.size() && (int)start < vm["instances"].as<int>(); ++minibatch_counter) {
        size_t end = min(training_corpus.size(), start + minibatch_size);

        #pragma omp master
        {
          global_gradient.setZero();
        }

        gradient.setZero();
        Real lambda = config.l2_parameter*(end-start)/static_cast<Real>(training_corpus.size()); 

        #pragma omp barrier
        cache_data(start, end, training_corpus, training_indices, training_instances);
        Real f = sgd_gradient(model, training_corpus, training_instances, lambda, g_R, g_Q, g_C, g_B);

        #pragma omp critical 
        {
          global_gradient += gradient;
          av_f += f;
        }
        #pragma omp barrier 
        #pragma omp master
        {
          adaGrad.array() += global_gradient.array().square();
          for (int w=0; w<model.num_weights(); ++w)
            if (adaGrad(w)) model.W(w) -= (step_size*global_gradient(w) / sqrt(adaGrad(w)));
						
					//cerr<<"l2-norm:"<<sqrt(model.W.array().square().sum())<<endl;
            
          // regularisation
          if (lambda > 0) av_f += (0.5*lambda*model.l2_gradient_update(step_size*lambda));

          if (minibatch_counter % 100 == 0) { cout << "."; cerr.flush(); }
        }

        //start += (minibatch_size*omp_get_num_threads());
        start += minibatch_size;
      }
      #pragma omp master
      cout << endl;

      Real iteration_time = (clock()-iteration_start) / (Real)CLOCKS_PER_SEC;
			Real perplexity_start = clock();
      if (vm.count("test-set")) {
        Real local_pp = perplexity(model, test_corpus, 1);
				//if (iteration == vm["iterations"].as<int>()-1) highestProbability(model, test_corpus);

        #pragma omp critical 
        { pp += local_pp; }
        #pragma omp barrier
      }

      #pragma omp master
      {
				Real perplexity_time = (clock()-perplexity_start) / (Real)CLOCKS_PER_SEC;
				
				cerr<<"pp:"<<pp<<endl;
        pp = exp(-pp/test_corpus.size());
        cout << " | Time: " << iteration_time << " seconds, Average f = " << av_f/training_corpus.size();
        if (vm.count("test-set")) {
          cout << ", Test Perplexity = " << pp<< ", Perplexity Time: " << perplexity_time << " seconds"; 
        }

        cout << " |" << endl << endl;
      }
    }
  }

  if (vm.count("model-out")) {
    cerr << "Writing trained model to " << vm["model-out"].as<string>() << endl;
    std::ofstream f(vm["model-out"].as<string>().c_str());
    boost::archive::text_oarchive ar(f);
    ar << model;
    cerr << "Finished writing trained model to " << vm["model-out"].as<string>() << endl;
  }
}


void cache_data(int start, int end, const Corpus& training_corpus, const vector<size_t>& indices, TrainingInstances &result) {
  assert (start>=0 && start < end && end <= static_cast<int>(training_corpus.size()));
  assert (training_corpus.size() == indices.size());

  size_t thread_num = omp_get_thread_num();
  size_t num_threads = omp_get_num_threads();

  result.clear();
  result.reserve((end-start)/num_threads);

  for (int s = start+thread_num; s < end; s += num_threads) {
    result.push_back(indices.at(s));
  }
}


Real sgd_gradient(HuffmanLogBiLinearModel& model,
                const Corpus& training_corpus,
                const TrainingInstances &training_instances,
                Real lambda, 
                LogBiLinearModel::WordVectorsType& g_R,
                LogBiLinearModel::WordVectorsType& g_Q,
                LogBiLinearModel::ContextTransformsType& g_C,
                LogBiLinearModel::WeightsType& g_B) {
  Real f=0;
  WordId start_id = model.label_set().Convert("<s>");
  WordId end_id = model.label_set().Convert("</s>");

  int word_width = model.config.word_representation_size;
  int context_width = model.config.ngram_order-1;

  // form matrices of the ngram histories
  clock_t cache_start = clock();
  int instances=training_instances.size();
  vector<MatrixReal> context_vectors(context_width, MatrixReal::Zero(instances, word_width)); 
  for (int instance=0; instance < instances; ++instance) {
    const TrainingInstance& t = training_instances.at(instance);
    int context_start = t - context_width;

    bool sentence_start = (t==0);
    for (int i=context_width-1; i>=0; --i) {
      int j=context_start+i;
      sentence_start = (sentence_start || j<0 || training_corpus.at(j) == end_id);
      int v_i = (sentence_start ? start_id : training_corpus.at(j));
      context_vectors.at(i).row(instance) = model.Q.row(v_i);
			
    }
  }
  MatrixReal prediction_vectors = MatrixReal::Zero(instances, word_width);
  for (int i=0; i<context_width; ++i){
    prediction_vectors += model.context_product(i, context_vectors.at(i));
	}
  
	clock_t cache_time = clock() - cache_start;

  MatrixReal nodeRepresentations = MatrixReal::Zero(instances, word_width);

  // calculate the function and gradient for each ngram

  clock_t iteration_start = clock();

	VectorReal rhat,R_gradient_contribution;
	
  for (int instance=0; instance < instances; instance++) {
	  int w_i = training_instances.at(instance);
    WordId w = training_corpus.at(w_i);

		double log_word_prob = 0;

    // do the gradient updates for R and B and compute objective function:
		for (int winstances=0;winstances<model.ys[w].size();winstances++){
			for (int i=model.ys[w][winstances].size()-2; i>=0;i--){
				int y=model.ys[w][winstances][i];
				int yIndex=model.ysInternalIndex[w][winstances][i+1];
				//cout<<"w:"<<model.label_str(w)<<" y:"<<y<<" yIndex:"<<yIndex<<endl;
				Real h = (model.R.row(yIndex) * prediction_vectors.row(instance).transpose()) + model.B(yIndex);
			
				//for computing objective function
				double binary_conditional_prob = ((y==1) ? log_sigmoid(h) : log_one_minus_sigmoid(h) ) ;		
						
				//for computing R and B gradients
				rhat=prediction_vectors.row(instance);
				//double gradientScalar=((y==1) ? ( (h>0) ? sigmoid(-h) : sigmoid(h) ) : (sigmoid(-h)-1) ) ; //gradient
				double gradientScalar=((y==1) ? ( -sigmoid(-h) ) : sigmoid(h) ) ; //negative gradient
				R_gradient_contribution = gradientScalar*rhat;
				double B_gradient_contribution = gradientScalar;

	      /////// Added for context gradient update /////// 
	  		nodeRepresentations.row(instance) += (gradientScalar*model.R.row(yIndex));
						 
				log_word_prob += binary_conditional_prob; //multiplying in log space
				g_R.row(yIndex) += R_gradient_contribution;
				//cout<<"g_B size:"<<g_B.size()<<" yindex:"<<yIndex<<endl;
				//cout<<"g_B(yIndex):"<<g_B(1)<<endl;
				g_B(yIndex) += B_gradient_contribution;
			}
		}
		//compute objective function
		assert(isfinite(log_word_prob));
		f -=log_word_prob;
		
  }
  clock_t iteration_time = clock() - iteration_start;

  clock_t context_start = clock();

  MatrixReal context_gradients = MatrixReal::Zero(word_width, instances);
  for (int i=0; i<context_width; ++i) {
    context_gradients = model.context_product(i, nodeRepresentations, true); // nodeRepresentations*C(i)^T

    for (int instance=0; instance < instances; ++instance) {
      int w_i = training_instances.at(instance);
      int j = w_i-context_width+i;

      bool sentence_start = (j<0);
      for (int k=j; !sentence_start && k < w_i; k++)
        if (training_corpus.at(k) == end_id) 
          sentence_start=true;
      int v_i = (sentence_start ? start_id : training_corpus.at(j));
			g_Q.row(v_i) += context_gradients.row(instance);						
    }
    model.context_gradient_update(g_C.at(i), context_vectors.at(i), nodeRepresentations);
  }
  
  clock_t context_time = clock() - context_start;

//	cerr<<"cache_time:"<<(cache_time/ (Real)CLOCKS_PER_SEC)<<" iteration_time:"<<(iteration_time/(Real)CLOCKS_PER_SEC)<<" context_time:"<<(context_time/ (Real)CLOCKS_PER_SEC)<<endl;

  return f;
}

void highestProbability(const HuffmanLogBiLinearModel& model, const Corpus& test_corpus){
	int word_width = model.config.word_representation_size;
  int context_width = model.config.ngram_order-1;

  int tokens=0;
  WordId start_id = model.label_set().Lookup("<s>");
  WordId end_id = model.label_set().Lookup("</s>");

	//fill context vectors
  vector<MatrixReal> context_vectors(context_width, MatrixReal::Zero(test_corpus.size(), word_width)); 
  for (int instance=0; instance < test_corpus.size(); ++instance) {
    const TrainingInstance& t = instance;
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

	//check if word has highest prob
	int biggerCount=0;
	for (int testIndex=0;testIndex<model.output_types();testIndex++){
		VectorReal probs=VectorReal::Zero(model.output_types());
		for (int w=0;w<model.output_types();w++){
	
			VectorReal prediction_vector = prediction_vectors.row(testIndex);
			double log_word_prob = getLogWordProb(model,prediction_vector,w);
			probs(w)=log_word_prob;
			//cerr<<"word:"<<model.label_str(w)<<" log_word_prob:"<<exp(log_word_prob)<<endl;
		} 
		int maxIndex=-1;
		probs.maxCoeff(&maxIndex);
		float epsilon=-5;
		if (probs(maxIndex) > (probs(testIndex)+epsilon) && maxIndex != testIndex){
			biggerCount++;
			cerr<<"actual word:"<<model.label_str(test_corpus.at(testIndex))<<" prob:"<<exp(probs(testIndex))<< " higher prob word:"<<model.label_str(test_corpus.at(maxIndex))<<" prob:"<<exp(probs(maxIndex))<<endl;
		}
	}
	cerr << "percent of words with a higher prob word:"<<biggerCount/model.output_types()<<endl;
}

Real perplexity(const HuffmanLogBiLinearModel& model, const Corpus& test_corpus, int stride) {
  Real p=0.0;

  int word_width = model.config.word_representation_size;
  int context_width = model.config.ngram_order-1;

  int tokens=0;
  WordId start_id = model.label_set().Lookup("<s>");
  WordId end_id = model.label_set().Lookup("</s>");

	//fill context vectors
  vector<MatrixReal> context_vectors(context_width, MatrixReal::Zero(test_corpus.size(), word_width)); 
  for (int instance=0; instance < test_corpus.size(); ++instance) {
    const TrainingInstance& t = instance;
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
	
  {
    #pragma omp master
    cout << "Calculating perplexity for " << test_corpus.size()/stride << " tokens"<<endl;

		// ofstream myfile;
		// 	  myfile.open ("distribution.txt");
		// myfile <<"[";
	
    size_t thread_num = omp_get_thread_num();
    size_t num_threads = omp_get_num_threads();
		VectorReal prediction_vector;
    for (size_t s = (thread_num*stride); s < test_corpus.size(); s += (num_threads*stride)) {
      WordId w = test_corpus.at(s);
			
			//get log of word probability
			prediction_vector = prediction_vectors.row(s);
			double log_word_prob = getLogWordProb(model,prediction_vector,w);
			//cerr<<"word prob "<<model.label_str(test_corpus.at(s))<<": "<<exp(log_word_prob)<<endl;
			
 			p += log_word_prob; //multiplying in log space
		  // myfile << model.unigram(w)<<","<<endl;
  		
      #pragma omp master
      if (tokens % 1000 == 0) { cout << "."; cout.flush(); }
  
      tokens++;
    }
    #pragma omp master
		// myfile<<"]";
		// myfile.close();
    cout << endl;
  }

  return p; 
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
