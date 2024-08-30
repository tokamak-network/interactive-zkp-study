// document.addEventListener(
//     'DOMContentLoaded', 
//     check_connection_change_interface(), 
//     false
//   );


async function test(){
    console.log("test!!");
}

async function save_code(){
    document.getElementById('code-form').submit();
    console.log("submitted");
}

async function delete_code(){
    document.getElementById('delete-form').submit();
    console.log("deleted");
}

async function ast_table(){
    document.getElementById('ast-form').submit();
    console.log("ast table created");
}

async function flatcode_table(){
    document.getElementById('flatcode-form').submit();
    console.log("flatcode table created");
}

async function abc_matrix(){
    document.getElementById('abc-matrix-form').submit();
    console.log("abc matrix created");
}

async function put_values(){
    document.getElementById('put-values-form').submit();
    console.log("retrieve inputs");
}

async function calculate_r_values(){
    document.getElementById('calculate-r-form').submit();
    console.log("calculate r");
}

async function create_qap(){
    document.getElementById('create-qap-form').submit();
    console.log("create qap");
}

async function create_qap_lcm(){
    document.getElementById('create-qap-lcm-form').submit();
    console.log("create qap lcm");
}

async function create_qap_fr(){
    document.getElementById('create-qap-fr-form').submit();
    console.log("create qap fr");
}