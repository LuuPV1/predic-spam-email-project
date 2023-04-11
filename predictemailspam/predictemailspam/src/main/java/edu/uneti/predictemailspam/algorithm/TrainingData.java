package edu.uneti.predictemailspam.algorithm;

import java.util.HashSet;
import java.util.Set;
import java.util.StringTokenizer;

public class TrainingData {
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
