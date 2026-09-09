use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::{LitStr, parse::Parse, parse_macro_input};

struct ProfileFunctionArgs {
	profiling_frame_placeholder: String,
}

impl Parse for ProfileFunctionArgs {
	fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
		let lit: LitStr = input.parse()?;
		let profiling_frame_placeholder = lit.value();
		Ok(Self {
			profiling_frame_placeholder,
		})
	}
}

/// creates a `ProfilingScope` for the entire function with the function's name
/// this proc macro requires one argument: a string literal that will be used to refer to the `ProfilingFrame`
/// whether the `ProfilingFrame` is a argument to the function (example: "#[profile_function("profiling_frame")]"),
/// or a field of `self` in a method (example: "#[profile_function("self.profiling_frame")]")
#[proc_macro_attribute]
pub fn profile_function(attr: TokenStream, input: TokenStream) -> TokenStream {
	let arg = parse_macro_input!(attr as ProfileFunctionArgs);

	let mut item: syn::Item = match syn::parse(input) {
		Ok(item) => item,
		Err(e) => {
			// forward compile error in the body of the function
			return e.into_compile_error().into();
		}
	};
	let fn_item = match &mut item {
		syn::Item::Fn(fn_item) => fn_item,
		_ => {
			return quote! {
				compile_error!("#[profile_function] is only for functions!"),
			}
			.into_token_stream()
			.into();
		}
	};
	let fn_name = fn_item.sig.ident.to_string();

	let profiling_frame_scope: syn::Stmt = match syn::parse_str(&format!(
		"crate::profile_scope!({}, \"{fn_name}\");",
		arg.profiling_frame_placeholder
	)) {
		Ok(profiling_frame_scope) => profiling_frame_scope,
		Err(e) => {
			let e_formatted = format!("{}", e);
			return quote! {
				compile_error!("supplied profiling_frame_placeholder is incorrect: {}", #e_formatted),
			}
			.into_token_stream()
			.into();
		}
	};

	fn_item.block.stmts.insert(0, profiling_frame_scope);

	item.into_token_stream().into()
}
