package edu.uneti.predictemailspam.service;

import edu.uneti.predictemailspam.model.entity.Email;

import java.io.IOException;
import java.util.List;

public interface EmailService {
    Email addEmail(Email email) throws IOException, ClassNotFoundException;

    List<Email> findAllEmail();

    Email findEmailById(Integer emailId);

    boolean delete(Integer emailId);
}
