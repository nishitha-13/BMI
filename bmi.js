alert("calculate bmi");
console.log(prompt("enter height and weight"));
let button=document.getElementById('btn');
button.addEventListener('click',()=>{
    const height =document.getElementById('height').value;
    const weight =document.getElementById('weight').value;
    const result =document.getElementById('output');
    let height_status=false, weight_status=false;
    if(height ==''|| isNaN(height) || (height<=0)){
        document.getElementById('height_error'.innerHTML ='please provide valide height');
    }else{
        document.getElementById('height_error').innerHTML='';
        height_status=true;
    }
    if(weight==''|| isNaN(weight) || (weight<=0)){
        document.getElementById('weight_error'.innerHTML ='please provide valide weight');
    }else{
        document.getElementById('weight_error').innerHTML='';
        weight_status=true;
    }
    if(height_status && weight_status){
        const bmi=(weight/((height*height)/10000)).toFixed(2);
        if(bmi<18.6){
            result.innerHTML='underweight: '+bmi;
        }else if(bmi>=18.6 && bmi<24.9){
            result.innerHTML='normal: ' +bmi;
        }else{
            result.innerHTML='overweight: '+bmi;

        }
    }else{
        alert('the form has errors');
        result.innerHTML='';
    }
    


});
