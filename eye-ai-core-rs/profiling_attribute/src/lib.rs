use proc_macro::TokenStream;
#[cfg(feature = "enable")]
use quote::quote;

#[cfg(feature = "enable")]
#[proc_macro_attribute]
pub fn profile_function(_args: TokenStream, input: TokenStream) -> TokenStream {
	let mut item: syn::Item = syn::parse(input).unwrap();
	let fn_item = match &mut item {
		syn::Item::Fn(fn_item) => fn_item,
		_ => panic!("expected fn"),
	};
	let fn_name = &fn_item.sig.ident.to_string();

	let profiling_attribute_code_at_top = quote!(
		let ___zone = tracing_tracy::client::span!(concat!(concat!(module_path!(), "::"), #fn_name));
	);

	fn_item.block.stmts.insert(
		0,
		syn::parse(profiling_attribute_code_at_top.into()).unwrap(),
	);

	use quote::ToTokens;
	item.into_token_stream().into()
}

#[cfg(not(feature = "enable"))]
#[proc_macro_attribute]
pub fn profile_function(_args: TokenStream, input: TokenStream) -> TokenStream {
	input
}
