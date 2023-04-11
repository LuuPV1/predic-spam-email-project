package edu.uneti.predictemailspam.algorithm;

import org.apache.commons.io.FileUtils;

import java.io.*;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;
import java.util.StringTokenizer;

public class FilterContentFile {
    // mảng chứa các túi từ của thư thường (non-spam)
    static ArrayList<Set<String>> listBagOfNonSpam = new ArrayList<>();
    // mảng chứa các túi từ của thư rác (spam)
    static ArrayList<Set<String>> listBagOfSpam = new ArrayList<>();

    // tinh xac xuat P(xi=x|nhan= nonspam)
    public static double pNonSpam(String x) {
        double k = 0;
        for (int i = 0; i < listBagOfNonSpam.size(); i++) {
            // moi lan x xuat hien trong 1 thu thuong thi k++
            if (listBagOfNonSpam.get(i).contains(x))
                k++;
        }
        return (k + 1) / (listBagOfNonSpam.size() + 1);
        // P(xi|nhan= nonspam)= (k+1)/(sothuthuong+1);
        // trong do: k la so cac mail nonspam xuat hien xi
        // sothuthuong la so mail nonspam
    }

    // tinh xac xuat P(xi=x|nhan= spam)
    public static double pSpam(String x) {
        double k = 0;
        for (int i = 0; i < listBagOfSpam.size(); i++) {
            if (listBagOfSpam.get(i).contains(x))
                // moi lan x xuat hien trong 1 thu rac thi k++
                k++;
        }
        return (k + 1) / (listBagOfSpam.size() + 1);
        // P(xi|nhan= spam)= (k+1)/(sothurac+1);
        // trong do: k la so cac mail spam xuat hien xi
        // sothurac la so mail spam
    }

    public boolean isSpamEmail(String content) throws IOException, ClassNotFoundException {
        training();
        // đọc dữ liệu huấn luyện từ trước ở trong file result_training.dat ra
        File file = new java.io.File("src/main/resources/static/result-training/result_training.dat");
        file.getParentFile().mkdirs(); // correct!
        if (!file.exists()) {
            file.createNewFile();
        }
        ObjectInputStream inp = new ObjectInputStream(
                new FileInputStream(file));
        listBagOfSpam = (ArrayList<Set<String>>) inp.readObject();
        listBagOfNonSpam = (ArrayList<Set<String>>) inp.readObject();
        inp.close();
        // đọc dữ liệu từ mail cần kiểm tra
        //File mailTesting = new File("C:\\Users\\admin\\Downloads\\lorealmail\\src\\main\\resources\\static\\test\\test (1).txt");
        // Tiền xử lý mail cần kiểm tra
        //String mailData = FileUtils.readFileToString(mailTesting, "UTF-16");
        Set<String> bagOfTest = TrainingData.toBagOfWord(content);
        // xác xuất là thư thường. P(xi|non-spam)
        double C_NB1 = listBagOfNonSpam.size() / ((double) listBagOfNonSpam.size() + listBagOfSpam.size());
        // xác xuất là thư rác. P(xi|spam)
        double C_NB2 = listBagOfSpam.size() / ((double) listBagOfNonSpam.size() + listBagOfSpam.size());
        ArrayList<String> listStringTest = new ArrayList<>(bagOfTest);
        for (String strTest : listStringTest) {
            if (pNonSpam(strTest) != ((double) 1 / (listBagOfNonSpam.size() + 1))
                    || pSpam(strTest) != ((double) 1 / (listBagOfSpam.size() + 1))) {
                C_NB1 *= pNonSpam(strTest);
                C_NB2 *= pSpam(strTest);
            }
        }
        if (C_NB1 < C_NB2) {
            // Bổ sung thư vừa kiểm tra vào tập huấn luyện.
            listBagOfSpam.add(bagOfTest);
        } else {
            listBagOfNonSpam.add(bagOfTest);
        }
        // Lưu lại tập huấn luyện mới.
        ObjectOutputStream out = new ObjectOutputStream(
                new FileOutputStream("src/main/resources/static/result-training/result_training.dat"));
        out.writeObject(listBagOfSpam);
        out.writeObject(listBagOfNonSpam);
        out.close();
        return C_NB1 < C_NB2;
    }

    private void training() throws IOException {
        // Đọc tất cả các file thư rác (spam), chuyển thành danh sách các túi từ
        File folderSpam = new File("src/main/resources/static/spam");
        ArrayList<Set<String>> listBagOfSpam = new ArrayList<>();
        File[] spams = folderSpam.listFiles();
        for (File file : spams) {
            String fileData = FileUtils.readFileToString(file, "UTF-16");
            Set<String> bagOfWord = toBagOfWord(fileData);
            listBagOfSpam.add(bagOfWord);
        }

        // Đọc tất cả các file thư thường (non-spam), chuyển thành danh sách các
        // túi từ
        File folderNonSpam = new File("src/main/resources/static/nonspam");
        ArrayList<Set<String>> listBagOfNonSpam = new ArrayList<>();
        File[] nonSpams = folderNonSpam.listFiles();
        for (File file : nonSpams) {
            String fileData = FileUtils.readFileToString(file, "UTF-16");
            Set<String> bagOfWord = toBagOfWord(fileData);
            listBagOfNonSpam.add(bagOfWord);
        }

        File file = new java.io.File("src/main/resources/static/result-training/result_training.dat");
        file.getParentFile().mkdirs(); // correct!
        if (!file.exists()) {
            file.createNewFile();
        }
        ObjectOutputStream out = new ObjectOutputStream(
                new FileOutputStream((file)));
        out.writeObject(listBagOfSpam);
        out.writeObject(listBagOfNonSpam);
        out.close();
        System.out.println("Hoàn thành huấn luyện");
    }

    public static Set<String> toBagOfWord(String s) {
        HashSet<String> bag = new HashSet<>();
        // tách các từ phân cách bởi các dấu ,.!*"'()
        StringTokenizer s1 = new StringTokenizer(s, " ,.!*\"\'()");
        while (s1.hasMoreTokens()) {
            bag.add(s1.nextToken());
        }
        return bag;
    }

}
