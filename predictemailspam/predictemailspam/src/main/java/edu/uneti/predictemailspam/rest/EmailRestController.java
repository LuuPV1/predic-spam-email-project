package edu.uneti.predictemailspam.rest;

import edu.uneti.predictemailspam.model.dto.EmailDto;
import edu.uneti.predictemailspam.model.entity.Email;
import edu.uneti.predictemailspam.model.mapper.EmailMapper;
import edu.uneti.predictemailspam.service.EmailService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.List;

@CrossOrigin("http://localhost:3000")
@RestController
@RequestMapping("/api/v1/email")
public class EmailRestController {
    @Autowired
    private EmailService emailService;
    @Autowired
    private EmailMapper emailMapper;

    @RequestMapping(value = "/emails", method = RequestMethod.GET)
    public ResponseEntity<List<EmailDto>> getAllEmail() {
        List<Email> emails = emailService.findAllEmail();
        List<EmailDto> res = emailMapper.toEmailList(emails);
        return new ResponseEntity<>(res, HttpStatus.OK);
    }

    @RequestMapping(value = "/detail-email", method = RequestMethod.GET)
    public ResponseEntity<EmailDto> getAllEmail(@RequestParam("emailId") Integer emailId) {
        Email email = emailService.findEmailById(emailId);
        EmailDto res = emailMapper.toDto(email);
        return new ResponseEntity<>(res, HttpStatus.OK);
    }

    @PostMapping(value = "/create")
    public ResponseEntity<EmailDto> createEmail(@RequestBody EmailDto email) throws IOException, ClassNotFoundException {
        LocalDate localDate = LocalDate.now();
        DateTimeFormatter formatters = DateTimeFormatter.ofPattern("dd/MM");
        String time = localDate.format(formatters);
        email.setTime(time);
        Email entity = emailMapper.toEmail(email);
        EmailDto res = emailMapper.toDto(emailService.addEmail(entity));
        return new ResponseEntity<>(res, HttpStatus.OK);
    }

    @DeleteMapping("/delete")
    public ResponseEntity<String> deleteEmail(@RequestParam("emailId") Integer emailId) {
        String status = emailService.delete(emailId) == true ? "Xoa email thanh cong" : "Co loi xay ra khi xoa email";
        return new ResponseEntity<>(status, HttpStatus.OK);
    }
}
