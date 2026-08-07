use jni::objects::{JByteArray, JObject, JString};
use jni::sys::{jlong, jstring};
use jni::JNIEnv;
use serde::Serialize;
use tokenizers::Tokenizer;

#[derive(Serialize)]
struct TokenizationResult {
    ids: Vec<u32>,
    attention_mask: Vec<u32>,
    token_type_ids: Vec<u32>,
}

fn throw_java_exception(env: &mut JNIEnv<'_>, class: &str, message: String) {
    let _ = env.throw_new(class, message);
}

fn parse_tokenizer(tokenizer_bytes: &[u8]) -> Result<Tokenizer, String> {
    Tokenizer::from_bytes(tokenizer_bytes)
        .map_err(|error| format!("Could not create tokenizer: {error}"))
}

#[no_mangle]
pub extern "system" fn Java_com_ml_shubham0204_sentence_1embeddings_HFTokenizer_createTokenizer(
    mut env: JNIEnv<'_>,
    _: JObject<'_>,
    tokenizer_bytes: JByteArray<'_>,
) -> jlong {
    let tokenizer_bytes = match env.convert_byte_array(&tokenizer_bytes) {
        Ok(bytes) => bytes,
        Err(error) => {
            throw_java_exception(
                &mut env,
                "java/lang/IllegalArgumentException",
                format!("Could not read tokenizer bytes: {error}"),
            );
            return 0;
        }
    };
    let tokenizer = match parse_tokenizer(&tokenizer_bytes) {
        Ok(tokenizer) => tokenizer,
        Err(message) => {
            throw_java_exception(&mut env, "java/lang/IllegalArgumentException", message);
            return 0;
        }
    };
    Box::into_raw(Box::new(tokenizer)) as jlong
}

#[no_mangle]
pub extern "system" fn Java_com_ml_shubham0204_sentence_1embeddings_HFTokenizer_tokenize(
    mut env: JNIEnv<'_>,
    _: JObject<'_>,
    tokenizer_ptr: jlong,
    text: JString<'_>,
) -> jstring {
    if tokenizer_ptr == 0 {
        throw_java_exception(
            &mut env,
            "java/lang/IllegalStateException",
            "Tokenizer is closed".to_owned(),
        );
        return std::ptr::null_mut();
    }
    let tokenizer = unsafe { &mut *(tokenizer_ptr as *mut Tokenizer) };
    let text: String = match env.get_string(&text) {
        Ok(text) => text.into(),
        Err(error) => {
            throw_java_exception(
                &mut env,
                "java/lang/IllegalArgumentException",
                format!("Could not read input text: {error}"),
            );
            return std::ptr::null_mut();
        }
    };
    let encoding = match tokenizer.encode(text, true) {
        Ok(encoding) => encoding,
        Err(error) => {
            throw_java_exception(
                &mut env,
                "java/lang/IllegalArgumentException",
                format!("Could not tokenize input text: {error}"),
            );
            return std::ptr::null_mut();
        }
    };
    let result = TokenizationResult {
        ids: encoding.get_ids().to_vec(),
        attention_mask: encoding.get_attention_mask().to_vec(),
        token_type_ids: encoding.get_type_ids().to_vec(),
    };
    let result_json = match serde_json::to_string(&result) {
        Ok(result_json) => result_json,
        Err(error) => {
            throw_java_exception(
                &mut env,
                "java/lang/IllegalStateException",
                format!("Could not serialize tokenization result: {error}"),
            );
            return std::ptr::null_mut();
        }
    };
    match env.new_string(result_json) {
        Ok(result) => result.into_raw(),
        Err(error) => {
            throw_java_exception(
                &mut env,
                "java/lang/IllegalStateException",
                format!("Could not return tokenization result: {error}"),
            );
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_com_ml_shubham0204_sentence_1embeddings_HFTokenizer_deleteTokenizer(
    _: JNIEnv,
    _: JObject,
    tokenizer_ptr: jlong,
) {
    if tokenizer_ptr != 0 {
        drop(unsafe { Box::from_raw(tokenizer_ptr as *mut Tokenizer) });
    }
}

#[cfg(test)]
mod tests {
    use super::parse_tokenizer;
    use tokenizers::models::bpe::BPE;
    use tokenizers::Tokenizer;

    #[test]
    fn parses_valid_tokenizer_bytes() {
        let tokenizer = Tokenizer::new(BPE::default());
        let tokenizer_json = tokenizer.to_string(false).expect("serialize tokenizer");

        assert!(parse_tokenizer(tokenizer_json.as_bytes()).is_ok());
    }

    #[test]
    fn rejects_invalid_tokenizer_bytes() {
        assert!(parse_tokenizer(b"not tokenizer json").is_err());
    }
}
