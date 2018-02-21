#include "../src/AI/util/IOData.hpp"
#include "../src/AI/util/Tensor.hpp"

int main(int argc, const char *argv[])
{
	
	ai::IOData test("test", 10);
	ai::Tensor_float test3(20);
	test3.fill(0, 1);
	test.pushNode("drop_proba", 0.435f);
	test.pushNode("input_size", 20);
	test.pushNode("label", "qwertyuio");
	test3.save(test, "weights");
	
	for (int i = 0; i < (int)test.getSubNodes().size(); i++)
		printf("%d %s\n", i, test.getSubNodes()[i].getName().c_str());
	
	printf("A1\n");
	test.writeToFile("test.txt");
	printf("A2\n");
	
	ai::IOData t2("");
	t2.loadFromFile("test.txt");
	printf("A3\n");
	printf("%s\n", t2.getName().c_str());
	int k;
	t2.get(k);
	printf("%s %d\n", t2.getName().c_str(), k);
	printf("%d\n", t2.existsNode("input_size"));
	printf("%d\n", t2.existsNode("drop_proba"));
	printf("%d\n", t2.existsNode("test"));
	for (int i = 0; i < (int)t2.getSubNodes().size(); i++)
		printf("%d %s\n", i, t2.getSubNodes()[i].getName().c_str());

	int is;
	t2.findNode("input_size")->get(is);
	printf("%d\n", is);
	float dp;
	t2.findNode("drop_proba")->get(dp);
	printf("%f\n", dp);
	std::string label;
	t2.findNode("label")->get(label);
	printf("%s\n", label.c_str());
	ai::Tensor_float test4(20);
	test4.load(t2, "weights");
	printf("%s\n", test4.tostring().c_str());
	printf("%d %d %d\n", test4.width(), test4.height(), test4.depth());
	return 0;
}
