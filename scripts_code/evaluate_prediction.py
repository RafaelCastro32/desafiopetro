import os

def read_line_as_integer(line):
    line = line.rstrip()
    line = line.rstrip(' ')
    print(line,'sssssss')
    number = int(line)
    return number

# reading files
num_errors = 0
num_predictions = 0
with open("./generate_dataset/test_labels.txt", "r") as f1:
    with open("./generate_dataset/predictions.txt", "r")  as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        N = len(lines1)
        if N != len(lines2):
            raise Exception('Error: files do not have the same number of lines:', N, len(lines2))
        for i in range(0,N,3):
            #first file
            number1_1 = int(lines1[i+1].rstrip())
            number1_2 = int(lines1[i+2].rstrip())
            #second file
            number2_1 = int(lines2[i+1].rstrip())
            number2_2 = int(lines2[i+2].rstrip())
            if number1_1 != number2_1:
                num_errors += 1
                print('Error in 1st digit of ', lines1[i].rstrip())
            if number1_2 != number2_2:
                num_errors += 1            
                print('Error in 2nd digit of ', lines1[i].rstrip())
            num_predictions += 2

print('Total number of predictions =',num_predictions)
print('Total number of errors =',num_errors)
error_rate = 100.0 * num_errors / num_predictions
print('Error rate = ', error_rate, '%')

def something():        
        i = 0
        line1 = f1.readline().rstrip(os.pathsep)
        i += 1
        print('aa',line1,'bb')
        line2 = f1.readline().rstrip()
        print('cca',line2,'bb')
        line3= f1.readline().rstrip()
        print('dd',line3,'bb')
        #print('aa',line,'bb')
        number = read_line_as_integer(line2)
        number = read_line_as_integer(f1)
        print(number,'Ak')
        
        for line2 in f2:
            
            # matching line1 from both files
            if line1 == line2:  
                # print IDENTICAL if similar
                print("Line ", i, ": IDENTICAL")       
            else:
                print("Line ", i, ":")
                # else print that line from both files
                print("\tFile 1:", line1, end='')
                print("\tFile 2:", line2, end='')
            break
  
# closing files
f1.close()                                       
f2.close()      
