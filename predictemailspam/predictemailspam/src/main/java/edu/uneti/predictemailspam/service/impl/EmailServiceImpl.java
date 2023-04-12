package edu.uneti.predictemailspam.service.impl;

import edu.uneti.predictemailspam.algorithm.FilterContentFile;
import edu.uneti.predictemailspam.algorithm.KNNFilter;
import edu.uneti.predictemailspam.model.entity.Email;
import edu.uneti.predictemailspam.repository.EmailRepository;
import edu.uneti.predictemailspam.service.EmailService;
import edu.uneti.predictemailspam.untils.ConverterString;
import jakarta.transaction.Transactional;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.List;

@Service
@Transactional
public class EmailServiceImpl implements EmailService {
    @Autowired
    private EmailRepository emailRepository;
    @Value("${use.algorithm}")
    private String algorithm;

    @Override
    public Email addEmail(Email email) throws IOException, ClassNotFoundException {
        //Check noi dung email va them co danh dau 1 email spam
        FilterContentFile fcf = new FilterContentFile();
        KNNFilter knn = new KNNFilter("src/main/resources/static/train/TrainingData.txt");
        boolean isSpam = false;
        if("knn".equals(algorithm)){
            isSpam = knn.predict(ConverterString.deAccent(email.getDescription()));
        }else{
            isSpam = fcf.isSpamEmail(email.getDescription());
        }
        email.setIsSpam(isSpam);
        return emailRepository.save(email);
    }

    @Override
    public List<Email> findAllEmail() {
        return emailRepository.findAll();
    }

    @Override
    public Email findEmailById(Integer emailId) {
        return emailRepository.findById(emailId).get();
    }

    @Override
    public boolean delete(Integer emailId) {
        Email email = emailRepository.findById(emailId).get();
        try{
            emailRepository.delete(email);
        }catch(Exception ex){
            return false;
        }
        return true;
    }
}