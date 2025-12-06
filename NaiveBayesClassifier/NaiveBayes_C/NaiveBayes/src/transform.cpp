#include <iostream>
#include <fstream>
#include <unordered_map>
#include <sstream>
#include <vector>

using namespace std;

unordered_map<string, unordered_map<string, int>> map =
{
	{ "É«Ôó", {{"ÇàÂÌ", 0}, {"ÎÚºÚ", 1}, {"Ç³°×", 2}} },
	{ "¸ùµÙ", {{"òéËõ", 0}, {"ÉÔòé", 1}, {"Ó²Í¦", 2}} },
	{ "ÇÃÉù", {{"×ÇÏì", 0}, {"³ÁÃÆ", 1}, {"Çå´à", 2}} },
	{ "ÎÆÀí", {{"ÇåÎú", 0}, {"ÉÔºı", 1}, {"Ä£ºı", 2}} },
	{ "Æê²¿", {{"°¼Ïİ", 0}, {"ÉÔ°¼", 1}, {"Æ½Ì¹", 2}} },
	{ "´¥¸Ğ", {{"Ó²»¬", 0}, {"ÈíÕ³", 1}} },
	{ "ºÃ¹Ï", {{"ÊÇ", 0}, {"·ñ", 1}} }
};

vector<string> getVec(string line)
{
	stringstream ss(line);
	string token;
	vector<string> portion;
	while (getline(ss, token, ','))
			portion.push_back(token);
	if (line.back() == ',')
		portion.push_back("");
	return portion;
}

//#define TRANSFORM
#ifdef TRANSFORM
int main()
{
	ifstream data("source/native_dataset.txt", ios::in);
	ifstream test("source/native_testset.txt", ios::in);
	ofstream out_data("source/encoded_dataset.txt", ios::out);
	ofstream out_test("source/encoded_testset.txt", ios::out);
	if (!data.is_open() || !test.is_open() || !out_data.is_open() || !out_test.is_open())
		return 1;

	string line;
	getline(data, line);
	vector<string> keylist = getVec(line);
	while (getline(data, line))
	{
		vector<string> portion = getVec(line);
		out_data << map["É«Ôó"][portion[0]] << ",";
		out_data << map["¸ùµÙ"][portion[1]] << ",";
		out_data << map["ÇÃÉù"][portion[2]] << ",";
		out_data << map["ÎÆÀí"][portion[3]] << ",";
		out_data << map["Æê²¿"][portion[4]] << ",";
		out_data << map["´¥¸Ğ"][portion[5]] << ",";
		out_data << portion[6] << ",";
		out_data << portion[7] << ",";
		out_data << map["ºÃ¹Ï"][portion[8]] << endl;
	}

	getline(test, line);
	while (getline(test, line))
	{
		vector<string> portion = getVec(line);
		out_test << map["É«Ôó"][portion[0]] << ",";
		out_test << map["¸ùµÙ"][portion[1]] << ",";
		out_test << map["ÇÃÉù"][portion[2]] << ",";
		out_test << map["ÎÆÀí"][portion[3]] << ",";
		out_test << map["Æê²¿"][portion[4]] << ",";
		out_test << map["´¥¸Ğ"][portion[5]] << ",";
		out_test << portion[6] << ",";
		out_test << portion[7] << ",";
		out_test << portion[8] << endl;
	}


	return 0;
}
#endif