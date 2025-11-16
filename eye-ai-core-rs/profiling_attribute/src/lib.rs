use proc_macro::TokenStream;
use quote::ToTokens;
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

	let mut item: syn::Item = syn::parse(input).unwrap();
	let fn_item = match &mut item {
		syn::Item::Fn(fn_item) => fn_item,
		_ => panic!("expected fn"),
	};
	let fn_name = fn_item.sig.ident.to_string();

	let profiling_frame_scope: syn::Stmt = syn::parse_str(&format!(
		"let _scope = {}.scope(\"{}\");",
		arg.profiling_frame_placeholder, fn_name
	))
	.unwrap();

	#[cfg(feature = "enable_tracy")]
	let tracy_span_code_at_top = quote!(
		let ___zone = tracing_tracy::client::span!(concat!(concat!(module_path!(), "::"), #fn_name));
	);

	fn_item.block.stmts.insert(0, profiling_frame_scope);

	#[cfg(feature = "enable_tracy")]
	fn_item
		.block
		.stmts
		.insert(0, syn::parse(tracy_span_code_at_top.into()).unwrap());

	item.into_token_stream().into()
}
